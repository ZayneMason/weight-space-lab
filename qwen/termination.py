#!/usr/bin/env python3
"""Is p50c a math edit or a stop-thinking edit? Paired closure analysis (roadmap #1).

Inputs (either source; nothing here runs a model or needs a GPU):

* ``upstream``: Takehi et al.'s released evaluation records, ``exp8_math/eval/<arm>/parts/
  math500_s{i}of10/``. Each shard holds per-problem correctness for every sample, shard
  aggregates, and the full text of sample 0 for its first 20 problems.
* ``local``: this lab's runner output directories (``job.json`` plus one JSON record per
  problem holding every generation, its text, finish reason and token IDs).

For every sample with text, the analysis records, from the text alone:

``closed``      the reasoning block was closed (``</think>`` is present);
``boxed``       a final ``\\boxed{}`` answer exists where the grader looks for it;
``correct``     the recorded grade (re-checked with this lab's grader);
``found``       some ``\\boxed{}`` anywhere in the text, reasoning included, verifies as the gold
                answer, i.e. the model wrote the right answer at some point;
``lenient``     no boxed answer, but a "Final Answer: X" / "the answer is X" phrase verifies;
``tail_ratio``  zlib-compressed / raw size of the last 20,000 characters (a repetition loop
                compresses far below 0.1);
``loop_onset``  where the maximal repetitive suffix starts, as a fraction of the text.

Samples of two arms are paired by problem (and sample index) and split into four cells by
whether each arm closed its reasoning. The accuracy difference between arms is exactly the
sum of the cells' contributions, so the table shows where an edit's gain comes from.
Problem-level bootstrap intervals keep every problem's samples together.

Token-budget curves are optional and need a tokenizer: each sample's token sequence is cut at
budget B, decoded and regraded. For local runs the stored token IDs are used directly; for
released text the text is re-encoded (approximate).

    python qwen/termination.py upstream /path/to/lm_understand-logs/exp8_math/eval runs/termination
    python qwen/termination.py local BASELINE_DIR P50C_DIR runs/termination-local --tokenizer tokenizer.json
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zlib

HERE = Path(__file__).resolve().parent
ARMS = ("unedited", "p10c", "p25c", "p50c")
CELLS = (("closed", "closed"), ("closed", "open"), ("open", "closed"), ("open", "open"))
DEFAULT_BUDGETS = (1024, 2048, 4096, 8192, 16384, 32768)
LOOP_THRESHOLD = 0.1
_GRADER = None


def grader():
    """The lab's MATH-500 grader (matches the released extractor); loaded lazily, never modified here."""
    global _GRADER
    if _GRADER is None:
        spec = importlib.util.spec_from_file_location("reproduce_qwen_grader", HERE / "reproduce_qwen.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GRADER = module
    return _GRADER


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ---------------------------------------------------------------- text measures

def boxed_at(text, at):
    """Content of the \\boxed at offset ``at``, parsed like the grader's last_boxed (brace-matched,
    or a space-separated form up to '$' or line end), without its '</think>' handling."""
    rest = text[at + len("\\boxed"):at + len("\\boxed") + 4000]
    value = rest.lstrip()
    if value.startswith("{"):
        depth = 0
        for i, char in enumerate(value):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                return value[1:i].strip() or None
        return None
    if rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def boxed_values(text, limit=64, max_scan=5000):
    """Distinct \\boxed contents anywhere in the text (reasoning included), in order of
    appearance, with their character offsets."""
    values, seen, start = [], set(), 0
    for _ in range(max_scan):
        at = text.find("\\boxed", start)
        if at < 0 or len(values) >= limit:
            break
        value = boxed_at(text, at)
        start = at + len("\\boxed")
        if value and value not in seen:
            seen.add(value)
            values.append((at, value))
    return values


def tail_ratio(text, window=20000):
    tail = text[-window:].encode("utf-8")
    return len(zlib.compress(tail, 6)) / len(tail) if tail else None


def loop_onset(text, window=4000, threshold=LOOP_THRESHOLD):
    """Start of the longest suffix made of windows that each compress below ``threshold``, as a
    fraction of the text length; None when the final window is not repetitive."""
    if len(text) < window:
        return None
    onset = None
    end = len(text)
    while end - window >= 0:
        chunk = text[end - window:end].encode("utf-8")
        if len(zlib.compress(chunk, 6)) / len(chunk) >= threshold:
            break
        onset = end - window
        end -= window
    return None if onset is None else onset / len(text)


LENIENT = (r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)",
           r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
           r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)")


def lenient_answer(text):
    """Latest 'Final Answer: X' / 'the answer is X' / 'Answer: X' phrase containing a digit or
    formula. An independent re-implementation of the released secondary (lenient) rule."""
    found = []
    for pattern in LENIENT:
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pattern, text)]
    for _, value in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", value):
            return value
    return None


def sample_features(text, gold, recorded_correct=None, grade=None, last_boxed=None):
    """Text-only measures of one generation. ``grade(text, gold)`` and ``last_boxed(text)`` default
    to the lab's grader; tests may inject simpler ones."""
    if grade is None or last_boxed is None:
        module = grader()
        grade = grade or (lambda t, g: module.grade(t, g)["correct"])
        last_boxed = last_boxed or module.last_boxed
    closed = "</think>" in text
    final = last_boxed(text)
    regraded = bool(grade(text, gold))
    found_at = None
    for at, value in boxed_values(text):
        if grade("\\boxed{" + value + "}", gold):
            found_at = at / max(len(text), 1)
            break
    lenient = None
    if final is None:
        answer_part = text.rsplit("</think>", 1)[1] if closed else text
        candidate = lenient_answer(answer_part)
        lenient = bool(candidate is not None and grade("\\boxed{" + candidate + "}", gold))
    return {"closed": closed, "boxed": final is not None,
            "correct": regraded if recorded_correct is None else bool(recorded_correct),
            "regrade_agrees": None if recorded_correct is None else regraded == bool(recorded_correct),
            "found": found_at is not None, "found_at": found_at, "lenient_correct": lenient,
            "tail_ratio": tail_ratio(text), "loop_onset": loop_onset(text), "chars": len(text),
            "text_sha256": sha(text)}


# ---------------------------------------------------------------- loading

def load_upstream(eval_dir, arm, shards=10):
    """Released MATH-500 records for one arm: aggregates, all-sample correctness, sample-0 texts."""
    root = Path(eval_dir) / arm / "parts"
    aggregate = {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0}
    problems, first, inputs = [], [], {}
    for shard in range(shards):
        folder = root / f"math500_s{shard}of{shards}"
        names = ("scores.json", "math500_per_problem.json", "math500_first_samples.jsonl.gz")
        for name in names:
            inputs[f"{arm}/parts/{folder.name}/{name}"] = file_sha(folder / name)
        scores = json.loads((folder / "scores.json").read_text(encoding="utf-8"))["math500"]
        for key in aggregate:
            aggregate[key] += scores[key]
        problems += json.loads((folder / "math500_per_problem.json").read_text(encoding="utf-8"))
        with gzip.open(folder / "math500_first_samples.jsonl.gz", "rt", encoding="utf-8") as stream:
            first += [json.loads(line) for line in stream if line.strip()]
    ids = [p["id"] for p in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{arm}: duplicate problem ids across shards")
    correct = {p["id"]: p["correct"] for p in problems}
    if sum(map(sum, correct.values())) != aggregate["ok"] or sum(map(len, correct.values())) != aggregate["total"]:
        raise ValueError(f"{arm}: per-problem correctness disagrees with shard aggregates")
    samples = [{"problem": r["id"], "sample": 0, "text": r["text"], "gold": r["answer"],
                "recorded_correct": correct[r["id"]][0], "token_ids": None, "finish_reason": None} for r in first]
    return {"arm": arm, "aggregate": aggregate, "correct": correct, "samples": samples, "inputs": inputs}


def load_local(run_dir):
    """This lab's runner output: every generation of every problem, with token IDs."""
    root = Path(run_dir)
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    correct, samples, aggregate = {}, [], {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0}
    for problem in job["problem_ids"]:
        record = json.loads((root / (sha(problem)[:20] + ".json")).read_text(encoding="utf-8"))
        if record["id"] != problem:
            raise ValueError("Problem record does not match its file name")
        correct[problem] = [int(g["correct"]) for g in record["generations"]]
        for g in record["generations"]:
            aggregate["ok"] += int(g["correct"])
            aggregate["total"] += 1
            aggregate["cut"] += g["finish_reason"] == "length"
            aggregate["tokens_sum"] += g["tokens"]
            samples.append({"problem": problem, "sample": g["sample"], "text": g["text"], "gold": record["answer"],
                            "recorded_correct": g["correct"], "token_ids": g.get("token_ids"), "finish_reason": g["finish_reason"]})
    return {"arm": job["arm"], "aggregate": aggregate, "correct": correct, "samples": samples,
            "inputs": {"job.json": file_sha(root / "job.json")}, "max_tokens": job["sampling"]["max_tokens"]}


# ---------------------------------------------------------------- statistics

def aggregate_bounds(aggregate):
    """What the aggregates alone imply: share of correct-and-capped samples is at least
    accuracy minus finish rate; accuracy among finished samples is at most accuracy / finish rate."""
    total = aggregate["total"]
    accuracy, finish = aggregate["ok"] / total, 1 - aggregate["cut"] / total
    return {"accuracy": accuracy, "cut_share": aggregate["cut"] / total, "finish_rate": finish,
            "mean_tokens": aggregate["tokens_sum"] / total,
            "correct_and_capped_at_least": max(0.0, accuracy - finish),
            "accuracy_given_finished_at_most": min(1.0, accuracy / finish) if finish else None}


def _bootstrap(values, statistic, replicates, seed):
    import numpy as np
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = statistic(values[index])
    return [float(x) for x in np.quantile(draws, [.025, .975], axis=0).T.reshape(-1)] if draws.ndim == 1 else \
        [[float(a), float(b)] for a, b in np.quantile(draws, [.025, .975], axis=0).T]


def accuracy_ci(correct, replicates=10000, seed=0):
    """Mean accuracy over all samples with a problem-level bootstrap interval."""
    import numpy as np
    problems = sorted(correct)
    rows = [[sum(correct[p]), len(correct[p])] for p in problems]
    arr = np.asarray(rows, dtype=float)
    ci = _bootstrap(arr, lambda d: d[..., 0].sum(-1) / d[..., 1].sum(-1), replicates, seed)
    return {"accuracy": float(arr[:, 0].sum() / arr[:, 1].sum()), "ci95": ci, "problems": len(rows), "samples": int(arr[:, 1].sum())}


def paired_accuracy(base, edit, replicates=10000, seed=0):
    """Accuracy difference over all samples of problems present in both arms (problem bootstrap)."""
    import numpy as np
    problems = sorted(set(base) & set(edit))
    arr = np.asarray([[np.mean(base[p]), np.mean(edit[p])] for p in problems])
    ci = _bootstrap(arr, lambda d: d[..., 1].mean(-1) - d[..., 0].mean(-1), replicates, seed)
    return {"problems": len(problems), "baseline": float(arr[:, 0].mean()), "edited": float(arr[:, 1].mean()),
            "delta": float(arr[:, 1].mean() - arr[:, 0].mean()), "delta_ci95": ci}


def paired_cells(base_features, edit_features, replicates=10000, seed=0):
    """Split paired samples by (baseline closed?, edited closed?) and attribute the accuracy difference.

    ``*_features`` map (problem, sample) to a feature dict. Contributions are
    (edited correct - baseline correct) / n per cell, so they sum to the total difference.
    """
    import numpy as np
    keys = sorted(set(base_features) & set(edit_features))
    if not keys:
        raise ValueError("No paired samples")
    names = [f"baseline_{a}/edited_{b}" for a, b in CELLS]
    rows = []
    for key in keys:
        b, e = base_features[key], edit_features[key]
        cell = names.index(f"baseline_{'closed' if b['closed'] else 'open'}/edited_{'closed' if e['closed'] else 'open'}")
        rows.append((cell, int(b["correct"]), int(e["correct"]), int(b["found"]), key[0]))
    n = len(rows)
    cells = {}
    for i, name in enumerate(names):
        members = [r for r in rows if r[0] == i]
        cells[name] = {"n": len(members), "baseline_correct": sum(r[1] for r in members), "edited_correct": sum(r[2] for r in members),
                       "baseline_found_answer": sum(r[3] for r in members),
                       "edited_correct_where_baseline_never_wrote_it": sum(r[2] and not r[3] for r in members),
                       "contribution": (sum(r[2] for r in members) - sum(r[1] for r in members)) / n}
    total = sum(c["contribution"] for c in cells.values())
    # Problem-clustered bootstrap over [cell one-hot (4), baseline correct, edited correct].
    problems = sorted({r[4] for r in rows})
    per_problem = np.zeros((len(problems), 10))
    where = {p: i for i, p in enumerate(problems)}
    for cell, bc, ec, _, problem in rows:
        row = per_problem[where[problem]]
        row[cell] += 1
        row[4 + cell] += ec - bc
        row[8] += 1
        row[9] += ec - bc

    def stats(d):
        count = d[..., 8].sum(-1)
        rescued = d[..., 4 + 2].sum(-1) / count
        both = d[..., 4 + 0].sum(-1) / np.maximum(d[..., 0].sum(-1), 1)
        return np.stack([d[..., 9].sum(-1) / count, rescued, both], axis=-1)

    ci = _bootstrap(per_problem, stats, replicates, seed)
    both = cells[names[0]]
    return {"pairs": n, "problems": len(problems), "cells": cells, "delta": total,
            "delta_ci95": ci[0], "rescued_cell_contribution_ci95": ci[1],
            "both_closed_accuracy_difference": (both["edited_correct"] - both["baseline_correct"]) / both["n"] if both["n"] else None,
            "both_closed_accuracy_difference_ci95": ci[2],
            "share_of_delta_from_baseline_open_edited_closed": cells[names[2]]["contribution"] / total if total else None}


def outcome_profile(features):
    """Shares of closed/boxed/correct/looping samples for one arm."""
    rows = list(features.values())
    n = len(rows)
    closed = [r for r in rows if r["closed"]]
    open_ = [r for r in rows if not r["closed"]]
    return {"samples": n, "closed": len(closed) / n, "accuracy": sum(r["correct"] for r in rows) / n,
            "accuracy_given_closed": sum(r["correct"] for r in closed) / len(closed) if closed else None,
            "correct_and_open": sum(r["correct"] for r in open_) / n,
            "closed_without_boxed_answer": sum(not r["boxed"] for r in closed) / n,
            "closed_without_box_but_lenient_correct": sum(bool(r["lenient_correct"]) for r in closed if not r["boxed"]) / n,
            "open_looping": sum(r["tail_ratio"] is not None and r["tail_ratio"] < LOOP_THRESHOLD for r in open_) / n,
            "open_looping_share_of_open": (sum(r["tail_ratio"] is not None and r["tail_ratio"] < LOOP_THRESHOLD for r in open_) / len(open_)) if open_ else None,
            "median_loop_onset_of_looping_open": _median([r["loop_onset"] for r in open_ if r["loop_onset"] is not None]),
            "found_answer_anywhere": sum(r["found"] for r in rows) / n,
            "regrade_agreement": _ratio([r["regrade_agrees"] for r in rows if r["regrade_agrees"] is not None])}


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _ratio(flags):
    return sum(flags) / len(flags) if flags else None


# ---------------------------------------------------------------- token budgets

def budget_curve(samples, budgets, decode, grade, encode=None, cap=32768):
    """Closure and accuracy if generation had stopped at each budget.

    Uses each sample's ``token_ids`` when present, otherwise ``encode(text)`` (approximate:
    re-encoding released text need not reproduce the original IDs). A sample whose full length
    fits within B keeps its recorded outcome; longer samples are cut at B, decoded and regraded.
    """
    out = {}
    for budget in budgets:
        closed = correct = 0
        for s in samples:
            ids = s["token_ids"] if s.get("token_ids") is not None else encode(s["text"])
            if len(ids) <= budget or budget >= cap:
                text, ok = s["text"], bool(s["recorded_correct"]) if s.get("recorded_correct") is not None else grade(s["text"], s["gold"])
            else:
                text = decode(ids[:budget])
                ok = bool(grade(text, s["gold"]))
            closed += "</think>" in text
            correct += ok
        out[budget] = {"closed": closed / len(samples), "accuracy": correct / len(samples)}
    return out


# ---------------------------------------------------------------- reports

def features_for(arm_data, grade=None, last_boxed=None):
    return {(s["problem"], s["sample"]): sample_features(s["text"], s["gold"], s["recorded_correct"], grade, last_boxed)
            for s in arm_data["samples"]}


def sanitized(features):
    """Per-sample measures without text, answers or raw problem ids."""
    keep = ("closed", "boxed", "correct", "regrade_agrees", "found", "found_at", "lenient_correct", "tail_ratio", "loop_onset", "chars", "text_sha256")
    return [{"problem_id_sha256": sha(problem), "sample": sample, **{k: f[k] for k in keep}}
            for (problem, sample), f in sorted(features.items())]


def report(arms, baseline, replicates=10000, grade=None, last_boxed=None):
    features = {name: features_for(data, grade, last_boxed) for name, data in arms.items()}
    summary = {"arms": {}, "paired": {}}
    for name, data in arms.items():
        summary["arms"][name] = {"aggregate": data["aggregate"], "aggregate_implies": aggregate_bounds(data["aggregate"]),
                                 "all_samples_accuracy": accuracy_ci(data["correct"], replicates),
                                 "text_samples": outcome_profile(features[name])}
    for name in arms:
        if name == baseline:
            continue
        summary["paired"][name] = {"all_samples": paired_accuracy(arms[baseline]["correct"], arms[name]["correct"], replicates),
                                   "text_samples_by_closure": paired_cells(features[baseline], features[name], replicates)}
    return summary, features


def plot(summary, baseline, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "weight-space-lab"})
    except ImportError:
        return None
    arms = list(summary["arms"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    parts = [("correct, finished reasoning", "#2a6f97"), ("wrong, finished reasoning", "#9ecae1"),
             ("correct, never finished", "#6a994e"), ("never finished: repetition loop", "#c0392b"),
             ("never finished: other", "#f0a8a0")]
    for i, arm in enumerate(arms):
        t = summary["arms"][arm]["text_samples"]
        closed_correct = t["accuracy"] - t["correct_and_open"]
        values = [closed_correct, t["closed"] - closed_correct, t["correct_and_open"], t["open_looping"],
                  1 - t["closed"] - t["correct_and_open"] - t["open_looping"]]
        bottom = 0
        for (label, color), v in zip(parts, values):
            axes[0].bar(i, v, bottom=bottom, color=color, label=label if i == 0 else None, width=.65)
            bottom += v
    axes[0].set_xticks(range(len(arms)), arms)
    axes[0].set_ylabel("share of first samples (200 problems)")
    axes[0].set_title("Where each model's answers end up")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.12), ncol=2)
    edits = [a for a in arms if a != baseline]
    labels = ["both finished", "only original finished", "only edit finished", "neither finished"]
    colors = ["#2a6f97", "#9ecae1", "#6a994e", "#c0392b"]
    for j, (label, color) in enumerate(zip(labels, colors)):
        name = f"baseline_{CELLS[j][0]}/edited_{CELLS[j][1]}"
        values = [summary["paired"][e]["text_samples_by_closure"]["cells"][name]["contribution"] * 100 for e in edits]
        axes[1].bar([k + (j - 1.5) * .2 for k in range(len(edits))], values, .2, color=color, label=label)
    for k, e in enumerate(edits):
        axes[1].plot([k - .4, k + .4], [summary["paired"][e]["text_samples_by_closure"]["delta"] * 100] * 2, color="#333", lw=1.5)
    axes[1].axhline(0, color="#888", lw=.8)
    axes[1].set_xticks(range(len(edits)), edits)
    axes[1].set_ylabel("contribution to accuracy gain (points)")
    axes[1].set_title("Which problems the gain comes from (line = total)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, metadata={"Date": None})
    return str(path)


def _write(out, summary, features, inputs, source):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "weight-space-qwen-termination-v1", "source": source, "inputs_sha256": inputs,
               "loop_threshold_tail_ratio": LOOP_THRESHOLD, **summary,
               "limits": ("Closure is read from text ('</think>' present), not from the engine's finish reason. "
                          "Paired cells use the samples whose text is available. 'found' only counts \\boxed{} answers. "
                          "Bootstrap intervals resample problems.")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "samples.json").write_text(json.dumps({arm: sanitized(f) for arm, f in features.items()}, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upstream", help="Analyse the released exp8_math/eval records")
    up.add_argument("eval_dir", type=Path)
    up.add_argument("output", type=Path)
    up.add_argument("--arms", nargs="+", default=list(ARMS))
    up.add_argument("--baseline", default="unedited")
    up.add_argument("--replicates", type=int, default=10000)
    up.add_argument("--source-revision", help="git revision of the released records, recorded in the summary")
    local = sub.add_parser("local", help="Analyse this lab's paired runner outputs")
    local.add_argument("baseline_dir", type=Path)
    local.add_argument("edited_dir", type=Path)
    local.add_argument("output", type=Path)
    local.add_argument("--tokenizer", type=Path, help="tokenizer.json for budget curves (needs the 'tokenizers' package)")
    local.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    local.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.command == "upstream":
        if args.baseline not in args.arms:
            parser.error("--baseline must be one of --arms")
        arms = {arm: load_upstream(args.eval_dir, arm) for arm in args.arms}
        summary, features = report(arms, args.baseline, args.replicates)
        inputs = {k: v for data in arms.values() for k, v in data["inputs"].items()}
        source = {"kind": "upstream released records", "revision": args.source_revision, "baseline": args.baseline,
                  "text_samples": "sample 0 of the first 20 problems of each shard (problems 0-199 of MATH-500)"}
    else:
        arms = {"baseline": load_local(args.baseline_dir), "p50c": load_local(args.edited_dir)}
        summary, features = report(arms, "baseline", args.replicates)
        inputs = {f"{name}/{k}": v for name, data in arms.items() for k, v in data["inputs"].items()}
        source = {"kind": "local runner outputs", "baseline": "baseline", "text_samples": "every generation"}
        if args.tokenizer:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(args.tokenizer))
            module = grader()
            summary["budget_curves"] = {name: budget_curve(data["samples"], args.budgets, lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                                                           lambda t, g: module.grade(t, g)["correct"], cap=data["max_tokens"])
                                        for name, data in arms.items()}
            inputs["tokenizer.json"] = file_sha(args.tokenizer)
    result = _write(args.output, summary, features, inputs, source)
    result["figure"] = plot(result, source["baseline"], Path(args.output) / "termination.svg")
    print(json.dumps({"output": str(args.output), "figure": result["figure"]}, indent=2))


if __name__ == "__main__":
    main()
