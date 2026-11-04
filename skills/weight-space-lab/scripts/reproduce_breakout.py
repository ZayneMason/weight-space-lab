#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "minatar==1.0.15"]
# ///
"""Reproduce published MinAtar Breakout weights and test a fresh tool-free editor."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np

REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
PREFIX = "exp5_write_from_scratch/"
ASSETS = {
    "prompt.txt": (PREFIX + "runs/breakout/message.txt", "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876"),
    "published-answer.md": (PREFIX + "runs/breakout/reply/answer.md", "a30bdda1acef79b507498edcef7e48d4fff16aa85783741925da9d5d4f42faf0"),
    "published.npz": (PREFIX + "networks/breakout/hand.npz", "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"),
    "published-scores.json": (PREFIX + "eval/minatar_eval.json", "bc10c6f957af95cdd52ee4be085acd249046f6117dbdb8c735fd5ddb2195c47c"),
    "LICENSE": ("LICENSE", "395f7febedfda6b5498ca36cfe3344d3986659ce4385dd8b03d2d75604250a8e"),
    "LICENSE-MinAtar": (PREFIX + "LICENSE-MinAtar", "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"),
}
TENSORS = {
    "W1": ("actor/dense1/kernel", (600, 32)), "b1": ("actor/dense1/bias", (32,)),
    "W2": ("actor/dense2/kernel", (32, 32)), "b2": ("actor/dense2/bias", (32,)),
    "W3": ("actor/logits/kernel", (32, 6)), "b3": ("actor/logits/bias", (6,)),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fetch(output):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for name, (source, expected) in ASSETS.items():
        url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{REVISION}/{source}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {name}")
        (output / name).write_bytes(payload)
        records.append({"file": name, "source": url, "sha256": expected, "bytes": len(payload)})
    save_json(output / "sources.json", {"revision": REVISION, "files": records})
    return {"downloaded": len(records), "revision": REVISION}


def numeric(node):
    if isinstance(node, ast.Constant) and type(node.value) in (float, int):
        if not math.isfinite(node.value) or abs(node.value) > 1e6:
            raise ValueError("Values must be finite and <=1e6 in magnitude")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = numeric(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("Unary signs apply only to numbers")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List):
        return [numeric(value) for value in node.elts]
    raise ValueError("Only numeric literals and lists are allowed; no executable Python")


def index_value(node, shape):
    axes = node.elts if isinstance(node, ast.Tuple) else [node]
    if len(axes) != len(shape):
        raise ValueError("Specify one index or slice per tensor dimension")
    result = []
    for axis, size in zip(axes, shape):
        if isinstance(axis, ast.Slice):
            parts = [numeric(v) if v is not None else d
                     for v, d in zip((axis.lower, axis.upper, axis.step), (0, size, 1))]
            start, stop, step = parts
            if any(type(v) is not int for v in parts) or not 0 <= start < stop <= size or step <= 0:
                raise ValueError("Slice must be nonempty, in bounds, with positive integer step")
            result.append(slice(start, stop, step))
        else:
            value = numeric(axis)
            if type(value) is not int or not 0 <= value < size:
                raise ValueError("Index outside tensor")
            result.append(value)
    return tuple(result)


def parse_edits(text):
    blocks = re.findall(r"^```edits\s*\n(.*?)^```\s*$", text, re.M | re.S)
    if not blocks:
        raise ValueError("No edits block found")
    tree = ast.parse(blocks[-1])
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, (_, shape) in TENSORS.items()}
    if not tree.body:
        raise ValueError("Empty edits block")
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, operator = statement.targets[0], "set"
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.op, (ast.Add, ast.Sub, ast.Mult)):
            target, operator = statement.target, type(statement.op).__name__
        else:
            raise ValueError(f"Unsupported statement at line {statement.lineno}")
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in arrays:
            raise ValueError("Only W1, b1, W2, b2, W3, b3 assignments are allowed")
        array = arrays[target.value.id]
        selection = index_value(target.slice, array.shape)
        value = np.asarray(numeric(statement.value), dtype=np.float32)
        selected = array[selection]
        if value.ndim and value.shape != np.shape(selected):
            raise ValueError("List shape must match assignment target")
        with np.errstate(over="raise", invalid="raise"):
            if operator == "set":
                array[selection] = value
            elif operator == "Add":
                array[selection] += value
            elif operator == "Sub":
                array[selection] -= value
            else:
                array[selection] *= value
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Non-finite result")
    return {TENSORS[name][0]: value for name, value in arrays.items()}, len(tree.body)


def import_answer(answer, output):
    arrays, statements = parse_edits(Path(answer).read_text())
    output = Path(output)
    with output.open("xb") as stream:
        np.savez(stream, **arrays)
    return {"statements": statements, "nonzero": sum(int(np.count_nonzero(a)) for a in arrays.values()),
            "answer_sha256": sha(answer), "checkpoint_sha256": sha(output)}


def load_weights(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {name for name, _ in TENSORS.values()}:
            raise ValueError("Expected exactly the six published policy tensors")
        arrays = []
        for key, shape in TENSORS.values():
            array = archive[key]
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid tensor: {key}")
            arrays.append(array)
        return arrays


def verify_reference(reference):
    reference = Path(reference)
    for name, (_, expected) in ASSETS.items():
        if sha(reference / name) != expected:
            raise ValueError(f"Asset modified: {name}")
    provenance = json.loads((reference / "sources.json").read_text(encoding="utf-8"))
    if set(provenance) != {"revision", "files"} or provenance["revision"] != REVISION or len(provenance["files"]) != len(ASSETS):
        raise ValueError("Reference provenance differs from the pinned asset inventory; fetch a new reference")
    seen = set()
    for row in provenance["files"]:
        if set(row) != {"file", "source", "sha256", "bytes"} or row["file"] not in ASSETS or row["file"] in seen:
            raise ValueError("Reference provenance has unexpected or duplicate fields/files")
        seen.add(row["file"])
        source, expected = ASSETS[row["file"]]
        url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{REVISION}/{source}"
        if row["source"] != url or row["sha256"] != expected or row["bytes"] != (reference / row["file"]).stat().st_size:
            raise ValueError("Reference provenance URL/hash/size differs from the pinned research assets")
    parsed, count = parse_edits((reference / "published-answer.md").read_text())
    published = load_weights(reference / "published.npz")
    for (name, _), weights in zip(TENSORS.values(), published):
        if not np.array_equal(parsed[name], weights):
            raise ValueError(f"Parsed response differs from published weights: {name}")
    return {"all_assets_verified": True, "exact_tensor_match": True, "statements": count,
            "parameters": sum(a.size for a in published), "revision": REVISION}


def observation(state):
    x = np.zeros((10, 10, 6), dtype=np.float32)
    x[:, :, 0] = state[:, :, 0]  # paddle -> player
    x[:, :, 2] = state[:, :, 3]  # bricks -> reward
    x[:, :, 3] = state[:, :, 1]  # ball -> projectile
    x[:, :, 4] = state[:, :, 2]  # previous ball -> trail
    return x.reshape(600)


def policy_logits(x, arrays):
    w1, b1, w2, b2, w3, b3 = arrays
    return np.tanh(np.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3


def init_worker(checkpoint):
    global WEIGHTS
    WEIGHTS = load_weights(checkpoint)


def episode(task):
    return run_episode(task, WEIGHTS)[0]


def run_episode(task, arrays, capture_frames=0):
    """Use the same seeded policy/environment loop for evaluation and preview."""
    from minatar import Environment
    index, seed, max_steps = task
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed + index)
    env.reset()
    rng = np.random.default_rng(seed + index + 1_000_000)
    reward = 0
    terminal = False
    frames = []
    for step in range(1, max_steps + 1):
        logits = policy_logits(observation(env.state()), arrays)
        probabilities = np.exp(logits.astype(np.float64) - logits.max())
        probabilities /= probabilities.sum()
        action = int(rng.choice(6, p=probabilities))
        r, terminal = env.act(action)
        reward += r
        if len(frames) < capture_frames:
            state = env.state().astype(np.uint8)
            bits = (state * np.array([1, 2, 4, 8], dtype=np.uint8)).sum(axis=2)
            frames.append({"step": step, "cells": ''.join(format(int(x), 'x') for x in bits.flat),
                           "action": action, "reward": int(r), "score": int(reward)})
        if terminal:
            break
    return {"episode": index, "score": reward, "steps": step, "truncated": not terminal}, frames


def validate_report(report):
    """Check stored statistics against complete, ordered episode evidence."""
    if report.get("schema") != "weight-space-breakout-v1" or report.get("game") != "breakout":
        raise ValueError("Expected a MinAtar Breakout evaluation report")
    required = ("settings", "versions", "checkpoint_sha256", "evaluator_sha256")
    for name in required:
        if name not in report:
            raise ValueError(f"Missing Breakout evidence field: {name}")
    settings = report["settings"]
    if not isinstance(settings, dict) or type(settings.get("max_steps")) is not int or settings["max_steps"] < 1:
        raise ValueError("Breakout settings require a positive integer max_steps")
    n = report.get("episodes")
    rows = report.get("results", [])
    if type(n) is not int or n < 2 or len(rows) != n:
        raise ValueError("Incomplete episode records; rerun the evaluation in a new output directory")
    if [r.get("episode") for r in rows] != list(range(n)):
        raise ValueError("Episode IDs must be unique, complete, and ordered from zero")
    limit = settings["max_steps"]
    for row in rows:
        if (type(row.get("score")) not in (int, float) or not math.isfinite(row["score"]) or row["score"] < 0
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= limit
                or type(row.get("truncated")) is not bool or (row["truncated"] and row["steps"] != limit)):
            raise ValueError("Invalid score, step count, or truncation flag in episode evidence")
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    expected = {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / math.sqrt(n)),
                "mean_length": float(np.mean([r["steps"] for r in rows]))}
    expected["ci95_normal"] = [expected["mean"] - 1.96 * expected["se"], expected["mean"] + 1.96 * expected["se"]]
    for name, value in expected.items():
        try:
            agrees = np.allclose(report.get(name), value, rtol=1e-12, atol=1e-12)
        except (TypeError, ValueError):
            agrees = False
        if not agrees:
            raise ValueError(f"Reported {name} disagrees with episode evidence")
    return expected


def evaluate(checkpoint, output, episodes=1000, seed=0, workers=4, max_steps=10000):
    if episodes < 2 or workers < 1 or max_steps < 1 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Invalid episodes, workers, step limit, or seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    load_weights(checkpoint)
    checkpoint_hash = sha(checkpoint)
    evaluator_hash = sha(__file__)
    started = time.monotonic()
    tasks = [(i, seed, max_steps) for i in range(episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(checkpoint),)) as pool:
        for result in pool.map(episode, tasks, chunksize=5):
            rows.append(result)
            if len(rows) % 50 == 0:
                print(f"Completed {len(rows)}/{episodes} episodes", file=sys.stderr, flush=True)
    if sha(checkpoint) != checkpoint_hash or sha(__file__) != evaluator_hash:
        raise ValueError("Checkpoint or evaluator changed during evaluation")
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    mean, se = float(scores.mean()), float(scores.std(ddof=1) / math.sqrt(episodes))
    result = {"schema": "weight-space-breakout-v1", "checkpoint_sha256": checkpoint_hash,
              "scope": "full_protocol" if episodes == 1000 and max_steps == 10000 else "smoke_or_custom",
              "evaluator_sha256": evaluator_hash, "game": "breakout", "episodes": episodes,
              "mean": mean, "se": se, "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
              "mean_length": float(np.mean([r["steps"] for r in rows])),
              "settings": {"seed": seed, "max_steps": max_steps, "sticky_action_probability": 0.1,
                           "action_selection": "softmax sampled; float64 softmax, float32 dense network",
                           "seeds": "env=seed+episode; action=seed+episode+1000000; fresh env per episode"},
              "versions": {"numpy": np.__version__, "minatar": importlib.metadata.version("minatar"), "python": sys.version.split()[0]},
              "seconds": time.monotonic() - started, "results": rows,
              "fidelity": "Matches published architecture, mapping, sampling policy, stickiness and default episode/step counts. Original evaluation seeds/backend were not released; not bit-exact score replication."}
    validate_report(result)
    save_json(output, result)
    return {k: v for k, v in result.items() if k != "results"}


DISABLED = "code_mode collaboration_modes shell_tool unified_exec unified_exec_tty shell_snapshot view_image image_generation multi_agent apps plugins remote_plugin browser_use browser_use_external browser_use_full_cdp_access in_app_browser computer_use code_mode_host skill_search skill_mcp_dependency_install sleep_tool tool_suggest goals hooks worktrees workspace_dependencies".split()


def audit_events(path):
    completed, started, seen, disallowed, notices = False, False, 0, [], []
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == "turn.completed":
            completed = True
        if event.get("type") == "turn.started":
            started = True
        if event.get("type", "").startswith("item."):
            kind = event.get("item", {}).get("type")
            message = event.get("item", {}).get("message", "")
            seen += 1
            # The CLI emits these startup diagnostics as error-typed items.
            # Accept only known pre-turn notices, never execution errors.
            known_notice = message.startswith((
                "Ignoring unknown `features` requirement `ultrafast_mode` from requirements layers:",
                "Under-development features enabled: skip_host_skill_discovery.",
                "Code Mode is unavailable because code-mode host is disabled.",
            ))
            if kind == "error" and not started and known_notice:
                notices.append(message)
            elif kind not in ("reasoning", "agent_message"):
                disallowed.append(kind)
        elif event.get("type") not in ("thread.started", "turn.started", "turn.completed"):
            disallowed.append(event.get("type"))
    return {"tool_free": completed and seen > 0 and not disallowed,
            "completed": completed, "observed_items": seen, "unexpected_events": disallowed,
            "startup_notices": notices}


def finalize(output):
    """Audit an already completed run and parse its answer, without another model call."""
    output = Path(output)
    request = json.loads((output / "request.json").read_text())
    run = json.loads((output / "run.json").read_text())
    if run["returncode"] != 0 or sha(output / "prompt.txt") != request["prompt_sha256"]:
        raise ValueError("Run failed or prompt changed")
    audit = audit_events(output / "events.jsonl")
    if not audit["tool_free"]:
        raise ValueError("Run is not verifiably tool-free")
    parsed = import_answer(output / "answer.md", output / "fresh.npz")
    result = {"audit": audit, "parser_sha256": sha(__file__), **parsed}
    save_json(output / "finalized.json", result)
    return result


def fresh(reference, output, model="gpt-6-astra", effort="xhigh", timeout=1200):
    reference, output = Path(reference), Path(output).resolve()
    if sha(reference / "prompt.txt") != ASSETS["prompt.txt"][1]:
        raise ValueError("Fresh run requires the pinned original prompt")
    output.mkdir(parents=False, exist_ok=False)
    (output / "prompt.txt").write_bytes((reference / "prompt.txt").read_bytes())
    (output / "instructions.txt").write_text("Answer the user's request.\n")
    command = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
               "--sandbox", "read-only", "--model", model, "--json", "--cd", str(output),
               "--output-last-message", str(output / "answer.md")]
    config = {"model_reasoning_effort": effort, "approval_policy": "never", "web_search": "disabled",
              "model_instructions_file": str(output / "instructions.txt"), "project_doc_max_bytes": 0,
              "agents.enabled": False, "features.skip_host_skill_discovery": True}
    for key, value in config.items():
        command.extend(["-c", key + "=" + json.dumps(value)])
    for feature in DISABLED:
        command.extend(["-c", f"features.{feature}=false"])
    command.append("-")
    record = {"model": model, "effort": effort, "prompt_sha256": sha(output / "prompt.txt"),
              "source_revision": REVISION, "command": command,
              "protocol": "One fresh response, no feedback. No published answer, weights, or scores in prompt."}
    save_json(output / "request.json", record)
    started = time.monotonic()
    with (output / "events.jsonl").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        run = subprocess.run(command, input=(output / "prompt.txt").read_text(), text=True,
                             stdout=stdout, stderr=stderr, timeout=timeout, check=False)
    record.update(returncode=run.returncode, seconds=time.monotonic() - started)
    record["audit"] = audit_events(output / "events.jsonl")
    save_json(output / "run.json", record)
    if run.returncode or not record["audit"]["tool_free"]:
        raise ValueError("Fresh run failed or was not verifiably tool-free; inspect run.json and stderr.log")
    parsed = import_answer(output / "answer.md", output / "fresh.npz")
    save_json(output / "parse.json", parsed)
    return {"model": model, "effort": effort, "audit": record["audit"], "seconds": record["seconds"], **parsed}


def compare(reference, replay, candidate=None):
    reference, replay = Path(reference), Path(replay)
    verify_reference(reference)
    published = json.loads((reference / "published-scores.json").read_text())["breakout"]
    r = json.loads(replay.read_text())
    validate_report(r)
    if r["schema"] != "weight-space-breakout-v1" or r["checkpoint_sha256"] != ASSETS["published.npz"][1]:
        raise ValueError("Replay is not an evaluation of the pinned published checkpoint")
    expected = published["hand"]
    result = {"published_mean": expected["mean"], "published_se": expected["se"],
              "replayed_mean": r["mean"], "replayed_se": r["se"], "episodes": r["episodes"],
              "difference": r["mean"] - expected["mean"],
              "difference_in_combined_standard_errors": (r["mean"] - expected["mean"]) / math.hypot(r["se"], expected["se"]),
              "published_ppo100_best": published["ppo100_s2"]["mean"],
              "published_ppo500_best": published["ppo500_s0"]["mean"],
              "ppo_status": "Published reference scores only; PPO checkpoints were not released or rerun."}
    full = r["episodes"] == 1000 and r["settings"]["max_steps"] == 10000
    result["scope"] = "full_protocol" if full else "smoke_or_custom"
    if not full:
        result.update(difference=None, difference_in_combined_standard_errors=None,
                      interpretation="Smoke/custom run: the published full-evaluation scores are context only; episode/step counts differ.")
    if candidate:
        c = json.loads(Path(candidate).read_text())
        validate_report(c)
        for field in ("schema", "game", "episodes", "settings", "versions", "evaluator_sha256"):
            if c[field] != r[field]:
                raise ValueError(f"Candidate evaluation differs in {field}. Re-evaluate both checkpoints with identical settings, versions, and evaluator code; previous files were preserved.")
        if len(c["results"]) != c["episodes"] or len(r["results"]) != r["episodes"]:
            raise ValueError("Incomplete episode records")
        if [x["episode"] for x in c["results"]] != [x["episode"] for x in r["results"]]:
            raise ValueError("Episode IDs differ")
        deltas = np.array([x["score"] - y["score"] for x, y in zip(c["results"], r["results"])])
        result.update(fresh_mean=c["mean"], fresh_se=c["se"], paired_delta=float(deltas.mean()),
                      paired_delta_se=float(deltas.std(ddof=1) / math.sqrt(len(deltas))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch")
    p.add_argument("output", type=Path)
    p = commands.add_parser("verify-reference")
    p.add_argument("reference", type=Path)
    p = commands.add_parser("import-answer")
    p.add_argument("answer", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("evaluate")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=10000)
    p = commands.add_parser("fresh")
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--effort", default="xhigh")
    p.add_argument("--timeout", type=int, default=1200)
    p = commands.add_parser("finalize")
    p.add_argument("output", type=Path)
    p = commands.add_parser("compare")
    p.add_argument("reference", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--candidate", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command").replace("-", "_")
    try:
        result = globals()[command](**args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, SyntaxError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
