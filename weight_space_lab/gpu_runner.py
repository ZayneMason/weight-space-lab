"""Single-host GPU job orchestration with immutable plans and owned-process cleanup."""
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from . import gpu_runtime as gpu
from .controller import _event, _save
from .evidence import digest, new_directory, read_json, write_json
from .legacy import ROOT

MODES = {"smoke": (2, 256, True), "pilot": (2, 32768, True), "full": (500, 32768, False)}
CODE = ("lab.py", "qwen/reproduce_qwen.py", "qwen/protocol.resolved.json", "weight_space_lab/__init__.py",
        "weight_space_lab/gpu_runtime.py", "weight_space_lab/gpu_runner.py", "weight_space_lab/controller.py",
        "weight_space_lab/evidence.py", "weight_space_lab/legacy.py", "LICENSE", "NOTICE.md", "licenses/upstream-research.txt")


def hardware(worker):
    return sorted((d["name"], d["memory_total_mib"], d["driver_version"]) for d in worker["devices"])


def plan_pair(root, output, config=None, mode="smoke", devices=None, python=None, eager=False,
              timeout_seconds=3600, allow_hardware_difference=False):
    if mode not in MODES or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 604800:
        raise ValueError("Choose smoke/pilot/full and a worker timeout of 1..604800 seconds")
    root, output = Path(root).resolve(), Path(output).resolve()
    layout = gpu.resolve(config or {}, devices)
    workers = layout["workers"]
    limit, max_tokens, pilot = MODES[mode]
    tasks = []
    for number, arm in enumerate(("baseline", "p50c")):
        worker = workers[number % len(workers)] if workers else None
        args = [str(python or sys.executable), str(ROOT / "qwen/reproduce_qwen.py"), "run", str(root), arm, str(output / arm),
                "--limit", str(limit), "--max-tokens", str(max_tokens), "--gpu-memory", str(layout["config"]["memory_fraction"]),
                "--tensor-parallel-size", str(layout["config"]["tensor_parallel_size"]),
                "--pipeline-parallel-size", str(layout["config"]["pipeline_parallel_size"]),
                "--max-num-seqs", str(layout["config"]["max_num_seqs"]), "--batch-problems", str(layout["config"]["batch_problems"]),
                "--placement-manifest", str(output / "manifest.json"), "--worker-id", worker["id"] if worker else "unassigned"]
        if pilot:
            args.append("--pilot")
        if eager:
            args.append("--eager")
        tasks.append({"id": arm, "arm": arm, "worker_id": worker["id"] if worker else None, "argv": args})
    matched = not workers or hardware(workers[0]) == hardware(workers[(1 if len(workers) > 1 else 0)])
    reasons = []
    if not layout["ready"]:
        reasons.append("GPU layout is not ready; inspect worker/skipped reasons")
    if not (root / "prepared.json").is_file():
        reasons.append("Prepared Qwen checkpoint directory is missing prepared.json")
    if output.exists():
        reasons.append("Output already exists; use a new directory")
    if not matched and not allow_hardware_difference:
        reasons.append("Arms would use different GPU classes; explicitly allow a hardware-different comparison or select matched groups")
    return {"schema": "weight-space-gpu-run-v1", "adapter": "qwen-paired", "root": str(root), "output": str(output),
            "mode": mode, "layout": layout, "tasks": tasks, "timeout_seconds": timeout_seconds,
            "eager": bool(eager), "allow_hardware_difference": bool(allow_hardware_difference), "hardware_matched": matched,
            "prepared_sha256": digest(root / "prepared.json") if (root / "prepared.json").is_file() else None,
            "code_sha256": {name: digest(ROOT / name) for name in CODE},
            "sampling_bounds": {"problems_per_arm": limit, "samples_per_problem": 4, "max_new_tokens_per_sample": max_tokens,
                                "maximum_output_tokens": 2*limit*4*max_tokens, "pilot": pilot},
            "ready": not reasons, "reasons": reasons,
            "limits": "Planning launches no models. Execution uses local frozen checkpoints, preserves existing GPU processes and requires --execute. Runtime fit/TP/PP support remain engine/model dependent."}


def _safe(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("GPU artifact path escapes its run")
    return path


def verify_manifest(path, runtime=False):
    path = Path(path)
    plan = read_json(path)
    if plan.get("schema") != "weight-space-gpu-run-v1" or plan.get("adapter") != "qwen-paired":
        raise ValueError("Unsupported GPU run manifest")
    if set(plan['code_sha256']) != set(CODE):
        raise ValueError("GPU source snapshot is incomplete")
    assigned = [next(w for w in plan['layout']['workers'] if w['id'] == t['worker_id']) for t in plan['tasks']]
    matched = hardware(assigned[0]) == hardware(assigned[1])
    if matched != plan['hardware_matched'] or (not matched and not plan['allow_hardware_difference']):
        raise ValueError("GPU hardware comparison is incorrectly declared")
    for name, expected in plan["code_sha256"].items():
        if digest(_safe(path.parent / "code", name)) != expected:
            raise ValueError("Recorded GPU source snapshot changed")
        if runtime and digest(_safe(ROOT, name)) != expected:
            raise ValueError("GPU runner source changed; use recorded code or prepare a new run")
    return plan


def _check_job(plan, job, worker_id):
    task = next((t for t in plan["tasks"] if t["arm"] == job['arm']), None)
    worker = next((w for w in plan["layout"]["workers"] if w["id"] == worker_id), None)
    if (task is None or worker is None or task["worker_id"] != worker_id
            or job['environment'].get("CUDA_VISIBLE_DEVICES") != worker["visible_devices"]
            or job['environment'].get("CUDA_DEVICE_ORDER") != 'PCI_BUS_ID'):
        raise ValueError("Worker affinity differs from the declared GPU placement")
    config = plan["layout"]["config"]
    for key, wanted in (("tensor_parallel_size", config["tensor_parallel_size"]), ("pipeline_parallel_size", config["pipeline_parallel_size"]),
                        ("gpu_memory_utilization", config["memory_fraction"]), ("max_num_seqs", config["max_num_seqs"]),
                        ("enforce_eager", plan['eager'])):
        if job['engine'].get(key) != wanted:
            raise ValueError("Engine settings differ from the GPU plan")
    bounds = plan['sampling_bounds']
    if (job['batch_problems'] != config['batch_problems'] or job['pilot'] != bounds['pilot']
            or len(job['problem_ids']) != bounds['problems_per_arm'] or job['sampling'].get('n') != bounds['samples_per_problem']
            or job['sampling'].get('max_tokens') != bounds['max_new_tokens_per_sample']
            or job['prepared_sha256'] != plan['prepared_sha256']
            or job['script_sha256'] != plan['code_sha256']['qwen/reproduce_qwen.py']):
        raise ValueError("Job bounds or frozen inputs differ from the GPU plan")
    return worker


def job_placement(manifest, job, worker_id, root, output):
    plan = verify_manifest(manifest, runtime=True)
    if Path(root).resolve() != Path(plan['root']) or Path(output).resolve() != Path(plan['output']) / job['arm']:
        raise ValueError("Job paths differ from the GPU plan")
    worker = _check_job(plan, job, worker_id)
    return {"manifest_sha256": digest(manifest), "worker_id": worker_id, "devices": worker["devices"]}


def validate_pair_placement(jobs, manifest):
    plan = verify_manifest(manifest)
    for job in jobs:
        placement = job.get("placement", {})
        worker = _check_job(plan, job, placement.get('worker_id'))
        if (placement.get("manifest_sha256") != digest(manifest)
                or placement.get("devices") != worker["devices"] or job["environment"].get("CUDA_VISIBLE_DEVICES") != worker["visible_devices"]
                or job["prepared_sha256"] != plan["prepared_sha256"] or job["script_sha256"] != plan["code_sha256"]["qwen/reproduce_qwen.py"]):
            raise ValueError("Paired job differs from the declared GPU run")
    environments = [{k: v for k, v in job["environment"].items() if k != "CUDA_VISIBLE_DEVICES"} for job in jobs]
    if environments[0] != environments[1]:
        raise ValueError("Unmatched paired environment beyond declared GPU affinity")
    return {"manifest_sha256": digest(manifest), "hardware_matched": plan["hardware_matched"],
            "allow_hardware_difference": plan["allow_hardware_difference"],
            "interpretation": "GPU affinity differences were predeclared; placement is not proof of bitwise or latency equivalence."}


def stop_owned(process):
    if os.name == 'nt':
        if process.poll() is None:
            try:
                subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'], capture_output=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
    else:
        # A GPU worker can have engine descendants after its parent exits.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if os.name != 'nt':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute_jobs(tasks, layout, output, inventory_provider=gpu.inspect, timeout_seconds=3600, log_limit_bytes=64*1024*1024):
    """Trusted operator argv lists, not a model-facing arbitrary command tool."""
    output = Path(output)
    workers = {w['id']: w for w in layout['workers']}
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) for name in ids):
        raise ValueError("GPU task IDs must be unique safe names")
    if any(t['worker_id'] not in workers or not isinstance(t['argv'], list) or not t['argv'] or any(not isinstance(a,str) for a in t['argv']) for t in tasks):
        raise ValueError("GPU tasks require a declared worker and argument list")
    if type(layout['max_concurrent']) is not int or layout['max_concurrent'] < 1:
        raise ValueError('GPU scheduler requires a positive concurrency limit')
    state = {"schema":"weight-space-gpu-state-v1", "phase":"running", "tasks":{t['id']:{"status":"pending"} for t in tasks}, "events":[]}
    pending, active, logs = list(tasks), {}, []
    _save(output, state)
    try:
        while pending or active:
            used = {item['task']['worker_id'] for item in active.values()}
            for task in list(pending):
                if len(active) >= layout['max_concurrent']:
                    break
                if task['worker_id'] in used:
                    continue
                worker = workers[task['worker_id']]
                reasons = gpu.readiness(worker, inventory_provider()['devices'], layout['config'])
                if reasons:
                    raise RuntimeError('; '.join(reasons))
                state['tasks'][task['id']] = {"status":"starting", "worker_id":worker['id']}
                _event(output,state,task['id'],'started',worker_id=worker['id'])
                log = (output / (task['id']+'.log')).open('xb'); logs.append(log)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=worker['visible_devices'], CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1')
                process = subprocess.Popen(task['argv'], stdout=log, stderr=subprocess.STDOUT, env=env,
                    start_new_session=os.name!='nt', creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                active[task['id']] = {"process":process,"task":task,"started":time.monotonic(),"log":log}
                state['tasks'][task['id']].update(status='running',pid=process.pid)
                _save(output,state)
                used.add(worker['id']); pending.remove(task)
            for name, item in list(active.items()):
                if time.monotonic()-item['started'] > timeout_seconds:
                    raise RuntimeError(f"GPU task timed out: {name}")
                if (output/(name+'.log')).stat().st_size > log_limit_bytes:
                    raise RuntimeError(f"GPU log capture exceeded its limit: {name}")
                code = item['process'].poll()
                if code is None:
                    continue
                stop_owned(item['process'])
                item['log'].close()
                if (output/(name+'.log')).stat().st_size > log_limit_bytes:
                    raise RuntimeError(f"GPU log capture exceeded its limit: {name}")
                state['tasks'][name].update(status='complete' if code==0 else 'failed',returncode=code)
                _event(output,state,name,'complete' if code==0 else 'failed',returncode=code)
                del active[name]
                if code:
                    raise RuntimeError(f"GPU task failed: {name} (exit {code})")
            if active:
                time.sleep(.05)
        state['phase']='complete'; _save(output,state)
        return state
    except BaseException as exc:
        cleanup = []
        for name, item in active.items():
            try:
                stop_owned(item['process'])
            except Exception as error:
                cleanup.append(type(error).__name__)
            state['tasks'][name].update(status='stopped',returncode=item['process'].poll())
        # Cleanup must finish for every child even when artifact storage fails.
        for name in active:
            try:
                _event(output,state,name,'stopped')
            except Exception as error:
                cleanup.append('journal: '+type(error).__name__)
        state['phase']='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        try:
            _save(output,state)
            write_json(output/'failure.json',{"exception":type(exc).__name__,"detail":str(exc),"cleanup_errors":cleanup,
                "note":"Only this run's owned process groups were stopped. Preserve partial artifacts; launch a new run."})
        except OSError:
            pass  # Original failure remains authoritative if storage is unavailable.
        raise
    finally:
        for log in logs:
            if not log.closed:
                log.close()
        for task in tasks:
            path=output/(task['id']+'.log')
            if path.exists() and path.stat().st_size > log_limit_bytes:
                with path.open('r+b') as stream:
                    stream.truncate(log_limit_bytes)


def launch_pair(plan):
    if not plan['ready']:
        raise ValueError("GPU run is not ready; review plan reasons")
    if os.name != 'posix':
        raise ValueError("Execute the vLLM backend under Linux/WSL; planning works on Windows")
    for name, expected in plan['code_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError("GPU plan source changed")
    if digest(Path(plan['root'])/'prepared.json') != plan['prepared_sha256']:
        raise ValueError("Prepared checkpoint manifest changed since planning")
    out=new_directory(plan['output'])
    write_json(out/'manifest.json',plan)
    for name in plan['code_sha256']:
        target=_safe(out/'code',name); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    try:
        assigned = {t['worker_id'] for t in plan['tasks']}
        with gpu.device_locks([w for w in plan['layout']['workers'] if w['id'] in assigned]):
            state = execute_jobs(plan['tasks'],plan['layout'],out,timeout_seconds=plan['timeout_seconds'])
            state['phase'] = 'comparing'; _save(out, state)
            verify_manifest(out/'manifest.json',runtime=True)
            command=[plan['tasks'][0]['argv'][0],str(ROOT/'qwen/reproduce_qwen.py'),'compare',str(out/'baseline'),str(out/'p50c'),str(out/'comparison.json'),
                     '--placement-manifest',str(out/'manifest.json')]
            result=subprocess.run(command,capture_output=True,text=True,timeout=300,check=False)
            (out/'comparison.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:
                raise RuntimeError("GPU workers finished but the paired comparison failed; inspect comparison.log")
            state['phase'] = 'complete'; _save(out, state)
        write_json(out/'integrity.json',{"manifest_sha256":digest(out/'manifest.json'),
            "artifacts":{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
            "integrity":"Unsigned local artifact ledger; not an independent attestation."})
        return {"complete":True,"comparison":str(out/'comparison.json'),"manifest":str(out/'manifest.json')}
    except BaseException as exc:
        if (out/'state.json').exists():
            state = read_json(out/'state.json')
            state['phase'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            _save(out, state)
        if not (out/'failure.json').exists():
            write_json(out/'failure.json',{"exception":type(exc).__name__,"detail":str(exc)})
        raise


def audit(output):
    """Verify local integrity and placement; does not rerun GPU inference or grade text."""
    out = Path(output).resolve()
    if (out/'failure.json').exists():
        raise ValueError('GPU run contains a failure record')
    ledger = read_json(out/'integrity.json')
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p != out/'integrity.json'}
    if actual != set(ledger['artifacts']) or ledger['manifest_sha256'] != digest(out/'manifest.json'):
        raise ValueError('GPU artifact inventory or manifest changed')
    for name, expected in ledger['artifacts'].items():
        if digest(_safe(out, name)) != expected:
            raise ValueError(f'GPU artifact changed: {name}')
    verify_manifest(out/'manifest.json')
    state = read_json(out/'state.json')
    if state['phase'] != 'complete' or any(t['status'] != 'complete' for t in state['tasks'].values()):
        raise ValueError('GPU run did not complete')
    jobs = [read_json(out/arm/'job.json') for arm in ('baseline', 'p50c')]
    placement = validate_pair_placement(jobs, out/'manifest.json')
    if read_json(out/'comparison.json').get('execution_placement') != placement:
        raise ValueError('GPU comparison placement differs from its jobs')
    return {'valid': True, 'artifacts': len(actual), 'execution_placement': placement,
            'limits': 'Unsigned local ledger and placement validation; inference and grading are not re-executed.'}
