"""Cut the margin refine's cost and check the picks hold (roadmap #74).

#73 found that the margin refine (``warm_triples.refine``: Nelder-Mead on a hinge loss of the label margins) takes
87% of #71's stage, and #65's stage builds committees with the same functions. A one-case profile (seed 264) showed
why: the committee's continuous fits call it twice for every one of about 8,900 candidate triples, and each call
spends about 270 loss evaluations, each of which copies the whole network and recomputes it in Python.

This module gives two cheaper refines:

- ``exact``: the same search, with a loss that computes the same numbers in the same order without copying the
  network, and reuses the hidden layer when no hidden-layer weight is edited. Its results should be bit-identical.
- ``columns``: the same search, with a loss that recomputes only the hidden units an edit touches. The margin is
  ``sign * (H @ (w1 - w0) + (b1 - b0))``; every hidden unit no edit touches is summed once, so each step costs a few
  1,024-long vectors instead of two matrix products. It equals the original loss to rounding (about 1e-16), so the
  search can take a slightly different path.
- ``loose``: ``columns`` with looser stopping (``xatol`` 1e-4, ``fatol`` 1e-7, at most 300 iterations).

``install(kind)`` swaps one in for ``warm_triples.refine`` everywhere the repair calls it. ``exact`` is checked
call by call in the tests and saves only about 15%, so the 35-case runs are ``columns`` and ``loose``: #73's runner
reruns #71's stage on #73's 35 cases from the same labels with the swapped refine, and the score compares each run
with #73's: core-hours, the chooser's pick, the network it chose and the probe's line.

    python experiments/locate-repair/cut_refine.py --kind columns --output runs/cut-columns --workers 5 --only 250 ...
    python experiments/locate-repair/cut_refine.py --score runs/cut-columns runs/cut-loose
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import early_values as ev

wt = ev.tc.wt
ORIGINAL = wt.refine
OPTIONS = {"original": {"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800},
           "exact": {"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800},
           "columns": {"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800},
           "loose": {"xatol": 1e-4, "fatol": 1e-7, "maxiter": 300}}


def margin_loss(case, triple, start, scale, x, y):
    """The refine's loss as a function of the step z, computing ``warm_triples.signed_margins`` without copies."""
    a = {n: v.astype(np.float64) for n, v in case["arrays"].items()}
    names = [n for n, _ in triple]
    flat = [(a[n].reshape(-1), i) for n, i in triple]
    sign = 2 * y - 1
    hidden = None
    if "hidden_weight" not in names and "hidden_bias" not in names:
        hidden = np.maximum(x @ a["hidden_weight"].T + a["hidden_bias"], 0)

    def loss(z):
        for (arr, i), v in zip(flat, start + z * scale):
            arr[i] = np.float64(np.float32(v))
        h = hidden if hidden is not None else np.maximum(x @ a["hidden_weight"].T + a["hidden_bias"], 0)
        logits = h @ a["weight"].T + a["bias"]
        return float(np.maximum(wt.EPS - sign * (logits[:, 1] - logits[:, 0]), 0).mean())

    return loss


def column_loss(case, triple, start, scale, x, y):
    """The same loss, recomputing only the hidden units (columns of H) that an edit touches."""
    a = {n: v.astype(np.float64) for n, v in case["arrays"].items()}
    W, b, w, c = a["hidden_weight"], a["hidden_bias"], a["weight"], a["bias"]
    sign = (2 * y - 1).astype(np.float64)
    inner = {i // W.shape[1] if n == "hidden_weight" else i for n, i in triple if n in ("hidden_weight", "hidden_bias")}
    units = sorted(inner | {i % w.shape[1] for n, i in triple if n == "weight"})
    fixed = [j for j in range(W.shape[0]) if j not in units]
    H = np.maximum(x @ W.T + b, 0)
    base = H[:, fixed] @ (w[1, fixed] - w[0, fixed])
    Hu = np.ascontiguousarray(H[:, units])  # columns of the touched units; the inner ones are rewritten each step
    redo = [(k, j) for k, j in enumerate(units) if j in inner]
    x0, x1 = x[:, 0].copy(), x[:, 1].copy()
    flat = [(a[n].reshape(-1), i) for n, i in triple]

    def loss(z):
        for (arr, i), v in zip(flat, start + z * scale):
            arr[i] = np.float64(np.float32(v))
        for k, j in redo:
            Hu[:, k] = np.maximum(x0 * W[j, 0] + x1 * W[j, 1] + b[j], 0)
        m = sign * (base + Hu @ (w[1, units] - w[0, units]) + (c[1] - c[0]))
        return float(np.maximum(wt.EPS - m, 0).mean())

    return loss


def refine(case, x, y, triple, best, kind="exact"):
    """``warm_triples.refine`` with a faster loss and the stopping rule of ``kind``."""
    scale = np.array([case["scale"][n] for n, _ in triple])
    start = np.array([v for _, _, v in best[1]])
    loss = (margin_loss if kind == "exact" else column_loss)(case, triple, start, scale, x, y)
    result = minimize(loss, np.zeros(3), method="Nelder-Mead",
                      options={"initial_simplex": np.vstack([np.zeros(3), 0.02 * np.eye(3)]), **OPTIONS[kind]})
    CALLS["calls"] += 1
    CALLS["evaluations"] += int(result.nfev)
    return wt.score(case, x, y, triple, start + result.x * scale)


CALLS = {"calls": 0, "evaluations": 0}


def install(kind):
    """Swap the refine everywhere the repair reaches it (every module calls it as ``wt.refine``)."""
    wt.refine = ORIGINAL if kind == "original" else (lambda case, x, y, triple, best: refine(case, x, y, triple, best, kind))


KINDS = ("columns", "loose")


def run_seed(args):
    """#73's timed run of one case, with the refine swapped before the run, plus the refine's call counts."""
    import time_repair
    seed, out, kind = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    CALLS.update(calls=0, evaluations=0)
    time_repair.run_seed((seed, out))
    row = json.loads(path.read_text())
    row.update(kind=kind, refine_calls=CALLS["calls"], refine_evaluations=CALLS["evaluations"])
    path.write_text(json.dumps(row) + "\n")
    return seed


def compare(rows, before):
    """Each run against #73's rows on the same cases: cost, the pick, the chosen network and the probe's line."""
    from time_repair import PARTS
    from pick_fixes import LINE
    d = lambda r: r["candidates"][r["full"]["pick"]]["disagreement"]
    stage = lambda rs: sum(r["seconds"][k] for r in rs.values() for k in PARTS) / 3600
    refine = lambda rs: sum(r["steps"].get("margin refine", 0) for r in rs.values()) / 3600
    ref = {s: before[s] for s in rows}
    winner = lambda r: r["candidates"][r["full"]["pick"]]["winner"]
    return {"seeds": len(rows), "core_hours_stage": stage(rows), "core_hours_stage_73": stage(ref),
            "core_hours_refine": refine(rows), "core_hours_refine_73": refine(ref),
            "speedup_stage": stage(ref) / stage(rows),
            "refine_calls": sum(r["refine_calls"] for r in rows.values()),
            "evaluations_per_call": sum(r["refine_evaluations"] for r in rows.values()) / sum(r["refine_calls"] for r in rows.values()),
            "under_0.01%": sum(d(r) < LINE for r in rows.values()), "under_0.01%_73": sum(d(r) < LINE for r in ref.values()),
            "exact": sum(r["candidates"][r["full"]["pick"]]["faults_found"] == 3 for r in rows.values()),
            "exact_73": sum(r["candidates"][r["full"]["pick"]]["faults_found"] == 3 for r in ref.values()),
            "same_pick": sum(rows[s]["full"]["pick"] == ref[s]["full"]["pick"] for s in rows),
            "same_winner": sum(winner(rows[s]) == winner(ref[s]) for s in rows),
            "same_disagreement": sum(d(rows[s]) == d(ref[s]) for s in rows),
            "crossed": {"gained": sorted(s for s in rows if d(ref[s]) >= LINE > d(rows[s])),
                        "lost": sorted(s for s in rows if d(rows[s]) >= LINE > d(ref[s]))},
            "changed": {str(s): [d(ref[s]), d(rows[s])] for s in sorted(rows) if d(rows[s]) != d(ref[s])}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path, nargs="*")
    a = p.parse_args()
    if a.score:
        from wide_compare import EVIDENCE
        out = EVIDENCE / "cut-refine"
        out.mkdir(parents=True, exist_ok=True)
        before = {r["seed"]: r for r in json.loads((EVIDENCE / "time-repair" / "time-repair.json").read_text())}
        result = {}
        for folder in a.score:
            rows = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in sorted(before)
                    if (folder / f"case-{s}.json").exists()}
            kind = next(iter(rows.values()))["kind"]
            (out / f"{kind}.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
            result[kind] = compare(rows, before)
        (out / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    install(a.kind)  # before time_repair wraps the refine in its clock, so the clock times the swapped one
    import time_repair
    a.output.mkdir(parents=True, exist_ok=True)
    slow = {251, 258, 259, 261, 263, 265, 270, 279, 281, 296, 309, 312, 317, 328, 348}
    jobs = [(s, str(a.output), a.kind) for s in sorted(a.only or time_repair.SEEDS, key=lambda s: s not in slow)]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
