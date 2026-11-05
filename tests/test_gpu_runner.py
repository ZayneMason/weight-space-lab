"""GPU run planning and CPU-only process fixtures; never launch a model."""
import json
import copy
import os
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

from weight_space_lab import gpu_runner
from weight_space_lab.evidence import write_json
from weight_space_lab.evidence import digest
from tests.test_gpu_runtime import inventory


class GpuRunnerTests(unittest.TestCase):
    def setUp(self):
        mask = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        mask.start(); self.addCleanup(mask.stop)

    def test_plan_is_read_only_and_assigns_parallel_or_serial_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"; root.mkdir()
            write_json(root / "prepared.json", {"fixture": True})
            out = Path(folder) / "out"
            plan = gpu_runner.plan_pair(root, out, {"device_groups": [["0"], ["2"]]}, devices=inventory())
            self.assertTrue(plan["ready"])
            self.assertFalse(out.exists())
            self.assertEqual([t["worker_id"] for t in plan["tasks"]], ["worker-0", "worker-1"])
            self.assertIn("--tensor-parallel-size", plan["tasks"][0]["argv"])
            one = gpu_runner.plan_pair(root, out, {"max_workers": 1}, devices=inventory())
            self.assertEqual([t["worker_id"] for t in one["tasks"]], ["worker-0", "worker-0"])
            self.assertEqual(plan["sampling_bounds"]["maximum_output_tokens"], 4096)

    def test_busy_gpu_plan_never_starts_a_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"; root.mkdir()
            write_json(root / "prepared.json", {})
            plan = gpu_runner.plan_pair(root, Path(folder)/"out", {"device_groups": [["1"]]}, devices=inventory())
            with patch.object(gpu_runner.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "not ready"):
                    gpu_runner.launch_pair(plan)
            popen.assert_not_called()

    def recorded_pair(self, folder, config=None, devices=None, allow_hardware_difference=False):
        root = Path(folder)/'model'; root.mkdir()
        write_json(root/'prepared.json', {})
        out = Path(folder)/'out'
        plan = gpu_runner.plan_pair(root, out, config or {'device_groups':[['0'],['2']]}, devices=devices or inventory(),
                                    allow_hardware_difference=allow_hardware_difference)
        out.mkdir()
        write_json(out/'manifest.json', plan)
        for name in plan['code_sha256']:
            target = out/'code'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gpu_runner.ROOT/name, target)
        jobs = []
        from qwen import reproduce_qwen as q
        for task in plan['tasks']:
            worker = next(w for w in plan['layout']['workers'] if w['id'] == task['worker_id'])
            job = dict(arm=task['arm'], pilot=True, problem_ids=['a','b'], sampling={'n':4,'max_tokens':256},
                       protocol={}, prepared_sha256=plan['prepared_sha256'], script_sha256=plan['code_sha256']['qwen/reproduce_qwen.py'],
                       versions={}, engine=q.engine_settings(), batch_problems=4,
                       environment={'CUDA_VISIBLE_DEVICES':worker['visible_devices'], 'CUDA_DEVICE_ORDER':'PCI_BUS_ID'})
            job['placement'] = gpu_runner.job_placement(out/'manifest.json', job, worker['id'], root, out/task['arm'])
            jobs.append(job)
        return out, plan, jobs

    def test_only_predeclared_affinity_can_differ_between_paired_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            out, plan, jobs = self.recorded_pair(folder)
            result = gpu_runner.validate_pair_placement(jobs, out/'manifest.json')
            self.assertTrue(result['hardware_matched'])
            for target, field, value in [('environment','VLLM_TEST','changed'), ('engine','max_num_seqs',8),
                                         ('sampling','max_tokens',32768), ('placement','worker_id','worker-0')]:
                changed = copy.deepcopy(jobs); changed[1][target][field] = value
                with self.assertRaises(ValueError):
                    gpu_runner.validate_pair_placement(changed, out/'manifest.json')
            changed = copy.deepcopy(jobs); changed[0]['batch_problems'] = 2
            with self.assertRaises(ValueError):
                gpu_runner.validate_pair_placement(changed, out/'manifest.json')
            with self.assertRaises(ValueError):
                gpu_runner.job_placement(out/'manifest.json', jobs[0], 'worker-0', Path(folder)/'wrong', out/'baseline')
            (out/'code/qwen/reproduce_qwen.py').write_text('changed')
            with self.assertRaises(ValueError):
                gpu_runner.validate_pair_placement(jobs, out/'manifest.json')

    def test_heterogeneous_hardware_requires_explicit_labelled_opt_in(self):
        devices = inventory(); devices[2]['name'] = 'another GPU'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'model'; root.mkdir(); write_json(root/'prepared.json', {})
            config = {'device_groups':[['0'],['2']]}
            plan = gpu_runner.plan_pair(root, Path(folder)/'out', config, devices=devices)
            self.assertFalse(plan['ready']); self.assertFalse(plan['hardware_matched'])
        with tempfile.TemporaryDirectory() as folder:
            out, plan, jobs = self.recorded_pair(folder, devices=devices, allow_hardware_difference=True)
            self.assertTrue(plan['ready'])
            result = gpu_runner.validate_pair_placement(jobs, out/'manifest.json')
            self.assertFalse(result['hardware_matched']); self.assertTrue(result['allow_hardware_difference'])

    def test_qwen_compare_requires_manifest_and_preserves_protocol_checks(self):
        import hashlib
        from qwen import reproduce_qwen as q
        with tempfile.TemporaryDirectory() as folder:
            out, plan, jobs = self.recorded_pair(folder)
            for job in jobs:
                arm = out/job['arm']; arm.mkdir(); write_json(arm/'job.json', job)
                for row_id in job['problem_ids']:
                    write_json(arm/(hashlib.sha256(row_id.encode()).hexdigest()[:20]+'.json'),
                               {'id':row_id,'generations':[{'correct':True,'truncated':False}]*4})
            with self.assertRaises(ValueError):
                q.compare(out/'baseline', out/'p50c', out/'comparison.json')
            q.compare(out/'baseline', out/'p50c', out/'comparison.json', out/'manifest.json')
            self.assertIn('execution_placement', json.loads((out/'comparison.json').read_text()))
            job = jobs[1]; job['sampling']['temperature'] = 1
            (out/'p50c/job.json').unlink(); write_json(out/'p50c/job.json', job)
            with self.assertRaisesRegex(ValueError, 'sampling'):
                q.compare(out/'baseline', out/'p50c', out/'bad.json', out/'manifest.json')

    def test_timeout_and_keyboard_interrupt_stop_owned_children(self):
        layout = gpu_runner.gpu.resolve({'device_groups':[['0'],['2']]}, inventory(), environ={})
        tasks = [{'id':str(i), 'worker_id':f'worker-{i}', 'argv':[sys.executable,'-c','import time;time.sleep(30)']} for i in range(2)]
        real_popen = gpu_runner.subprocess.Popen
        for interrupted in (False, True):
            children = []
            def spawn(*args, **kwargs):
                child = real_popen(*args, **kwargs); children.append(child); return child
            count = 0
            def snapshot():
                nonlocal count
                count += 1
                if interrupted and count == 2:
                    raise KeyboardInterrupt()
                return {'devices':inventory()}
            with tempfile.TemporaryDirectory() as folder, patch.object(gpu_runner.subprocess,'Popen',side_effect=spawn):
                with self.assertRaises(KeyboardInterrupt if interrupted else RuntimeError):
                    gpu_runner.execute_jobs(tasks, layout, Path(folder), inventory_provider=snapshot, timeout_seconds=.1)
                self.assertTrue(children)
                self.assertTrue(all(c.poll() is not None for c in children))
                state = json.loads((Path(folder)/'state.json').read_text())
                self.assertEqual(state['phase'], 'interrupted' if interrupted else 'failed')

    def test_completed_ledger_detects_changes_and_extra_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out, plan, jobs = self.recorded_pair(folder)
            for job in jobs:
                (out/job['arm']).mkdir(); write_json(out/job['arm']/'job.json', job)
            write_json(out/'comparison.json', {'execution_placement':gpu_runner.validate_pair_placement(jobs, out/'manifest.json')})
            write_json(out/'state.json', {'phase':'complete', 'tasks':{j['arm']:{'status':'complete'} for j in jobs}})
            write_json(out/'integrity.json', {'manifest_sha256':digest(out/'manifest.json'),
                       'artifacts':{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}})
            self.assertTrue(gpu_runner.audit(out)['valid'])
            (out/'extra').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                gpu_runner.audit(out)
            (out/'extra').unlink(); (out/'state.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                gpu_runner.audit(out)

    def test_cleanup_stops_every_child_even_if_journaling_fails(self):
        layout = gpu_runner.gpu.resolve({'device_groups':[['0'],['2']]}, inventory(), environ={})
        tasks = [{'id':str(i), 'worker_id':f'worker-{i}', 'argv':[sys.executable,'-c','import time;time.sleep(30)']} for i in range(2)]
        children = []
        real_popen, real_event = gpu_runner.subprocess.Popen, gpu_runner._event
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs); children.append(child); return child
        def journal(*args, **kwargs):
            if args[3] == 'stopped':
                raise OSError('fixture storage failure')
            return real_event(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch.object(gpu_runner.subprocess,'Popen',side_effect=spawn), patch.object(gpu_runner,'_event',side_effect=journal):
            try:
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    gpu_runner.execute_jobs(tasks, layout, Path(folder), inventory_provider=lambda:{'devices':inventory()}, timeout_seconds=.1)
                self.assertTrue(all(c.poll() is not None for c in children))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill(); child.wait()

    def test_shutdown_output_cannot_bypass_log_budget(self):
        from unittest.mock import Mock
        layout = gpu_runner.gpu.resolve({'max_workers':1}, inventory(), environ={})
        task = {'id':'a','worker_id':'worker-0','argv':['fixture']}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            child = Mock(pid=123); child.poll.return_value = 0
            def shutdown(process):
                (out/'a.log').write_bytes(b'x'*256)
            with patch.object(gpu_runner.subprocess,'Popen',return_value=child), patch.object(gpu_runner,'stop_owned',side_effect=shutdown):
                with self.assertRaisesRegex(RuntimeError, 'log capture exceeded'):
                    gpu_runner.execute_jobs([task], layout, out, inventory_provider=lambda:{'devices':inventory()}, log_limit_bytes=128)
            self.assertEqual((out/'a.log').stat().st_size, 128)
            self.assertEqual(json.loads((out/'state.json').read_text())['phase'], 'failed')
            self.assertTrue((out/'failure.json').exists())

    def test_single_worker_reuses_gpu_sequentially_with_uuid_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            layout = gpu_runner.gpu.resolve({"max_workers": 1}, inventory(), environ={})
            tasks = [{"id": arm, "worker_id": "worker-0", "argv": [sys.executable, "-c",
                     'import os,json;print(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))']} for arm in ("a", "b")]
            state = gpu_runner.execute_jobs(tasks, layout, out, inventory_provider=lambda: {"devices": inventory()}, timeout_seconds=10)
            self.assertEqual(state["phase"], "complete")
            self.assertEqual([e["outcome"] for e in state["events"]], ["started", "complete", "started", "complete"])
            self.assertEqual(json.loads((out/"a.log").read_text())["gpu"], "GPU-fixture-0")

    def test_concurrent_workers_have_separate_masks_and_failures_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            layout = gpu_runner.gpu.resolve({"device_groups": [["0"],["2"]]}, inventory(), environ={})
            tasks = [{"id": str(i), "worker_id": f"worker-{i}", "argv": [sys.executable,"-c","import time;time.sleep(.2)"]} for i in range(2)]
            state = gpu_runner.execute_jobs(tasks, layout, Path(folder), inventory_provider=lambda: {"devices": inventory()}, timeout_seconds=10)
            self.assertEqual([e["outcome"] for e in state["events"][:2]], ["started", "started"])
        with tempfile.TemporaryDirectory() as folder:
            tasks[0]["argv"] = [sys.executable,"-c","raise SystemExit(7)"]
            with self.assertRaises(RuntimeError):
                gpu_runner.execute_jobs(tasks, layout, Path(folder), inventory_provider=lambda: {"devices": inventory()}, timeout_seconds=10)
            state = json.loads((Path(folder)/"state.json").read_text())
            self.assertEqual(state["phase"], "failed")
            self.assertTrue((Path(folder)/"failure.json").exists())

    def test_memory_is_rechecked_before_launch_and_gpu_locks_exclude_second_owner(self):
        from weight_space_lab import gpu_runtime
        layout = gpu_runtime.resolve({"max_workers":1}, inventory(), environ={})
        with tempfile.TemporaryDirectory() as folder:
            with gpu_runtime.device_locks(layout["workers"], Path(folder)):
                with self.assertRaises(RuntimeError):
                    with gpu_runtime.device_locks(layout["workers"], Path(folder)):
                        pass
        with tempfile.TemporaryDirectory() as folder:
            changed = inventory(); changed[0]["memory_free_mib"] = 1
            with patch.object(gpu_runner.subprocess,"Popen") as popen:
                with self.assertRaises(RuntimeError):
                    gpu_runner.execute_jobs([{"id":"a","worker_id":"worker-0","argv":["unused"]}], layout, Path(folder),
                                            inventory_provider=lambda:{"devices":changed})
            popen.assert_not_called()
