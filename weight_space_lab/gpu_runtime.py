"""NVIDIA/CUDA inventory, explicit GPU groups and conservative launch readiness."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from contextlib import contextmanager

DEFAULTS = {"device_groups": "auto", "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "memory_fraction": .65, "headroom_mib": 1024, "max_workers": None,
            "execution": "parallel", "max_num_seqs": 16, "batch_problems": 4, "respect_visible_devices": True}


def parse_inventory(text):
    rows, indices, uuids = [], set(), set()
    for fields in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError("NVIDIA inventory must report index, UUID, name, total/free MiB and driver")
        index, uuid, name, total, free, driver = (f.strip() for f in fields)
        index, total, free = int(index), int(total), int(free)
        if (index < 0 or total <= 0 or not 0 <= free <= total or index in indices or uuid in uuids
                or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) or not name or not driver):
            raise ValueError("Inconsistent or duplicate NVIDIA inventory row")
        indices.add(index); uuids.add(uuid)
        rows.append({"index": index, "uuid": uuid, "name": name, "memory_total_mib": total,
                     "memory_free_mib": free, "driver_version": driver})
    return sorted(rows, key=lambda row: row["index"])


def inspect():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "devices": [], "reason": "NVIDIA inventory is unavailable; no GPU job was started"}
    if result.returncode:
        return {"available": False, "devices": [], "reason": "nvidia-smi failed; no GPU job was started"}
    return {"available": True, "devices": parse_inventory(result.stdout), "utc": datetime.now(timezone.utc).isoformat()}


def configuration(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown GPU configuration fields")
    config = {**DEFAULTS, **value}
    for key, maximum in (("tensor_parallel_size", 64), ("pipeline_parallel_size", 64), ("max_num_seqs", 256), ("batch_problems", 64)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if (type(config["memory_fraction"]) not in (float, int) or not math.isfinite(config["memory_fraction"])
            or not 0 < config["memory_fraction"] <= 1 or type(config["headroom_mib"]) is not int or config["headroom_mib"] < 0):
        raise ValueError("Use a finite memory_fraction in (0,1] and nonnegative headroom_mib")
    if config["max_workers"] is not None and (type(config["max_workers"]) is not int or not 1 <= config["max_workers"] <= 64):
        raise ValueError("max_workers must be null or 1..64")
    if config["execution"] not in ("parallel", "sequential"):
        raise ValueError("execution must be parallel or sequential")
    if type(config["respect_visible_devices"]) is not bool:
        raise ValueError("respect_visible_devices must be a boolean")
    if config["device_groups"] != "auto" and (not isinstance(config["device_groups"], list) or not config["device_groups"]):
        raise ValueError("device_groups must be auto or a nonempty list of disjoint device lists")
    return config


def _select(selector, devices):
    if type(selector) is int:
        selector = str(selector)
    if not isinstance(selector, str):
        raise ValueError("GPU selectors must be nvidia-smi indices or complete GPU UUIDs")
    matches = [d for d in devices if selector in (str(d["index"]), d["uuid"])]
    if len(matches) != 1:
        raise ValueError(f"GPU is unavailable or not visible: {selector}")
    return matches[0]


def visible_inventory(devices, environ):
    mask = environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return devices
    if mask in ("", "-1"):
        return []
    selected = [_select(s.strip(), devices) for s in mask.split(',')]
    if len({d["uuid"] for d in selected}) != len(selected):
        raise ValueError("CUDA_VISIBLE_DEVICES repeats a physical GPU")
    return selected


def readiness(worker, devices, config):
    current = {d["uuid"]: d for d in devices}
    reasons = []
    for expected in worker["devices"]:
        found = current.get(expected["uuid"])
        if found is None or any(found[k] != expected[k] for k in ("name", "memory_total_mib", "driver_version")):
            reasons.append(f"GPU identity changed or disappeared: {expected['uuid']}")
            continue
        required = math.ceil(found["memory_total_mib"] * config["memory_fraction"]) + config["headroom_mib"]
        if found["memory_free_mib"] < required:
            reasons.append(f"Insufficient free memory on GPU {found['index']}: {found['memory_free_mib']} MiB available, {required} MiB required")
    return reasons


def resolve(value, devices=None, environ=None):
    config = configuration(value)
    if devices is None:
        devices = inspect()["devices"]
    devices = visible_inventory(devices, os.environ if environ is None else environ) if config["respect_visible_devices"] else devices
    size = config["tensor_parallel_size"] * config["pipeline_parallel_size"]
    groups, skipped = [], []
    if config["device_groups"] == "auto":
        available = []
        for d in devices:
            reason = readiness({"devices": [d]}, devices, config)
            if reason:
                skipped.append({"index": d["index"], "uuid": d["uuid"], "reasons": reason})
            else:
                available.append(d)
        for offset in range(0, len(available), size):
            group = available[offset:offset+size]
            if len(group) == size:
                groups.append(group)
            else:
                skipped.extend({"index": d["index"], "uuid": d["uuid"], "reasons": ["Not enough eligible GPUs for another complete worker group"]} for d in group)
    else:
        for group in config["device_groups"]:
            if not isinstance(group, list) or len(group) != size:
                raise ValueError("Every device group must contain tensor_parallel_size × pipeline_parallel_size GPUs")
            groups.append([_select(selector, devices) for selector in group])
    flat = [d["uuid"] for group in groups for d in group]
    if len(flat) != len(set(flat)):
        raise ValueError("GPU worker groups must not overlap, including index/UUID aliases")
    if config["max_workers"] is not None:
        groups = groups[:config["max_workers"]]
    workers = []
    for i, group in enumerate(groups):
        reasons = readiness({"devices": group}, devices, config)
        workers.append({"id": f"worker-{i}", "devices": group, "visible_devices": ','.join(d["uuid"] for d in group),
                        "ready": not reasons, "reasons": reasons})
    return {"schema": "weight-space-gpu-layout-v1", "backend": "cuda", "config": config, "workers": workers,
            "skipped": skipped, "ready": bool(workers) and all(w["ready"] for w in workers),
            "max_concurrent": len(workers) if config["execution"] == "parallel" else min(1, len(workers)),
            "selector_policy": "nvidia-smi physical indices/full UUIDs; children use UUID masks and PCI_BUS_ID order",
            "limits": "Single-host CUDA placement. Memory checks are snapshots, not inference-fit certification or an OS reservation. No existing process is stopped."}


@contextmanager
def device_locks(workers, directory=None):
    root = Path(directory) if directory else Path(tempfile.gettempdir()) / "weight-space-lab-gpu-locks"
    root.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for uuid in sorted({d["uuid"] for w in workers for d in w["devices"]}):
            path = root / (hashlib.sha256(uuid.encode()).hexdigest() + '.lock')
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                raise RuntimeError("A selected GPU is locked by another lab run; investigate stale locks without deleting active reservations") from None
            held.append(path)
            with os.fdopen(fd, 'w') as stream:
                stream.write(str(os.getpid()))
        yield
    finally:
        for path in held:
            path.unlink()
