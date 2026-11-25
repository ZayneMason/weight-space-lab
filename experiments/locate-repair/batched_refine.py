"""One Nelder-Mead run over many triples at once, in arrays (roadmap #76).

#73 and #75 found the margin refine (``warm_triples.refine``: Nelder-Mead on a hinge loss of the label margins) is most
of the repair's cost, and #74's faster loss still leaves it 68% to 76% of each stage. Most of its calls come from the
committee, which refines every one of about 8,900 candidate triples twice, one Python loop and one SciPy run per
triple. This module runs all of a committee's refines together: SciPy's Nelder-Mead steps (non-adaptive, the same
initial simplex, ``xatol``, ``fatol`` and ``maxiter``), followed by every problem in lockstep, with the loss evaluated for
all the problems that need a point at each step in one array expression.

The loss is #74's columns loss: the output-margin contribution of every hidden unit no edit touches is summed once,
and only the (at most three) touched units are recomputed. Each problem keeps a small state vector holding the
touched units' input weights, biases and output weights and the output biases; an edit writes its float32-rounded
value into one slot, as ``warm_triples.signed_margins`` applies it.

``install()`` swaps ``triple_committee.committee`` for a copy whose continuous fits go through ``batch_refine``; every
other refine (the refits, the told triple) stays one call at a time, with the columns loss.

    python experiments/locate-repair/batched_refine.py --output runs/batched --workers 4 \
        --only 250 264 291 297 307 320 328 338
    python experiments/locate-repair/batched_refine.py --score runs/batched
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import early_values as ev
from wide_compare import EVIDENCE

tc = ev.tc
wt, ct = tc.wt, tc.ct
OPTIONS = {"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800}
CHUNK = 1500  # problems per array expression; 1,096 labels x 1,500 is about 13 MB per temporary
BLOCK = 3000  # problems set up and run together; their saved columns take about 80 MB
CALLS = {"calls": 0, "evaluations": 0}
RHO, CHI, PSI, SIGMA = 1, 2, 0.5, 0.5
# state slots: 0-5 the touched units' input weights (unit, input), 6-8 their biases, 9-11 and 12-14 their weights
# into output 1 and output 0, 15 and 16 the output biases
W_IN, B_IN, W1, W0, C1, C0 = 0, 6, 9, 12, 15, 16


def setup(case, x, y, jobs):
    """Per problem: the constant margin part, the state vector at the start values, and the slot each edit writes."""
    a = {n: v.astype(np.float64) for n, v in case["arrays"].items()}
    W, b, w, c = a["hidden_weight"], a["hidden_bias"], a["weight"], a["bias"]
    H = np.maximum(x @ W.T + b, 0)
    dw = w[1] - w[0]
    total = H @ dw
    width = W.shape[0]
    K = len(jobs)
    base = np.empty((K, len(x)))
    state = np.zeros((K, 17))
    slot = np.empty((K, 3), int)
    start = np.empty((K, 3))
    scale = np.empty((K, 3))
    fixed = np.zeros((K, 3, len(x)))  # the saved hidden column of each touched unit whose own inputs stay put
    inner = np.zeros((K, 3), bool)  # touched units whose input weights or bias are edited, recomputed each step
    for k, (triple, best) in enumerate(jobs):
        def unit(n, i):
            return i // W.shape[1] if n == "hidden_weight" else i % width if n == "weight" else i if n == "hidden_bias" else None
        units = sorted({unit(n, i) for n, i in triple} - {None})
        base[k] = total - H[:, units] @ dw[units]
        for u, j in enumerate(units):
            state[k, W_IN + 2 * u:W_IN + 2 * u + 2] = W[j]
            state[k, B_IN + u], state[k, W1 + u], state[k, W0 + u] = b[j], w[1, j], w[0, j]
        state[k, C1], state[k, C0] = c[1], c[0]
        for p, (n, i) in enumerate(triple):
            if n == "bias":
                slot[k, p] = C1 if i == 1 else C0
                continue
            u = units.index(unit(n, i))
            slot[k, p] = (W_IN + 2 * u + i % W.shape[1] if n == "hidden_weight" else B_IN + u if n == "hidden_bias"
                          else (W1 if i // width == 1 else W0) + u)
        for n, i in triple:
            if n in ("hidden_weight", "hidden_bias"):
                inner[k, units.index(unit(n, i))] = True
        for u, j in enumerate(units):
            if not inner[k, u]:
                fixed[k, u] = H[:, j]
        start[k] = [v for _, _, v in best[1]]
        scale[k] = [case["scale"][n] for n, _ in triple]
    return base, state, slot, start, scale, fixed, inner


def losses(problem, rows, z, x, sign):
    """The hinge loss of ``z`` (one step per row) for the problems ``rows``. Units whose input weights and bias no edit
    touches keep their saved hidden column; only the edited ("inner") units are recomputed, one slot at a time."""
    base, state, slot, start, scale, fixed, inner = problem
    out = np.empty(len(rows))
    x0, x1 = x[:, 0], x[:, 1]
    for lo in range(0, len(rows), CHUNK):
        r, zz = rows[lo:lo + CHUNK], z[lo:lo + CHUNK]
        s = state[r].copy()
        values = (start[r] + zz * scale[r]).astype(np.float32).astype(np.float64)
        np.put_along_axis(s, slot[r], values, axis=1)
        dw = s[:, W1:W1 + 3] - s[:, W0:W0 + 3]
        m = base[r] + (s[:, C1] - s[:, C0])[:, None]
        f = fixed[r]
        for u in range(3):
            m += f[:, u] * dw[:, u, None]
        inn = inner[r]
        for u in range(3):
            k = np.flatnonzero(inn[:, u])
            if len(k):
                h = x0 * s[k, W_IN + 2 * u, None]
                h += x1 * s[k, W_IN + 2 * u + 1, None]
                h += s[k, B_IN + u, None]
                np.maximum(h, 0, out=h)
                h *= dw[k, u, None]
                m[k] += h
        m *= sign
        np.subtract(wt.EPS, m, out=m)
        np.maximum(m, 0, out=m)
        out[lo:lo + CHUNK] = m.mean(axis=1)
    return out


def nelder_mead(f, K, xatol, fatol, maxiter):
    """SciPy's Nelder-Mead (non-adaptive, no bounds) on K problems in lockstep. ``f(rows, z)`` evaluates one step
    per row. Returns each problem's best step and its number of loss evaluations."""
    N = 3
    sim = np.broadcast_to(np.vstack([np.zeros(N), 0.02 * np.eye(N)]), (K, N + 1, N)).copy()
    all_rows = np.arange(K)
    fsim = f(np.repeat(all_rows, N + 1), sim.reshape(-1, N)).reshape(K, N + 1)
    calls = np.full(K, N + 1)
    order = np.argsort(fsim, axis=1, kind="stable")
    sim, fsim = np.take_along_axis(sim, order[:, :, None], 1), np.take_along_axis(fsim, order, 1)
    iterations = 1
    active = all_rows
    while len(active) and iterations < maxiter:
        s, fs = sim[active], fsim[active]
        done = ((np.abs(s[:, 1:] - s[:, :1]).reshape(len(active), -1).max(1) <= xatol)
                & (np.abs(fs[:, :1] - fs[:, 1:]).max(1) <= fatol))
        active, s, fs = active[~done], s[~done], fs[~done]
        if not len(active):
            break
        xbar = np.add.reduce(s[:, :-1], 1) / N
        worst = s[:, -1]
        xr = (1 + RHO) * xbar - RHO * worst
        fxr = f(active, xr)
        calls[active] += 1
        expand = fxr < fs[:, 0]
        accept_r = ~expand & (fxr < fs[:, -2])
        outside = ~expand & ~accept_r & (fxr < fs[:, -1])
        inside = ~expand & ~accept_r & ~outside
        second = np.empty_like(xr)
        second[expand] = (1 + RHO * CHI) * xbar[expand] - RHO * CHI * worst[expand]
        second[outside] = (1 + PSI * RHO) * xbar[outside] - PSI * RHO * worst[outside]
        second[inside] = (1 - PSI) * xbar[inside] + PSI * worst[inside]
        need = ~accept_r
        f2 = np.full(len(active), np.nan)
        f2[need] = f(active[need], second[need])
        calls[active[need]] += 1
        new, fnew = xr.copy(), fxr.copy()
        take2 = (expand & (f2 < fxr)) | (outside & (f2 <= fxr)) | (inside & (f2 < fs[:, -1]))
        new[take2], fnew[take2] = second[take2], f2[take2]
        shrink = (outside & ~(f2 <= fxr)) | (inside & ~(f2 < fs[:, -1]))
        keep = ~shrink
        s[keep, -1], fs[keep, -1] = new[keep], fnew[keep]
        if shrink.any():
            sh = s[shrink]
            sh[:, 1:] = sh[:, :1] + SIGMA * (sh[:, 1:] - sh[:, :1])
            rows = active[shrink]
            fs[shrink, 1:] = f(np.repeat(rows, N), sh[:, 1:].reshape(-1, N)).reshape(-1, N)
            calls[rows] += N
            s[shrink] = sh
        order = np.argsort(fs, axis=1, kind="stable")
        sim[active], fsim[active] = np.take_along_axis(s, order[:, :, None], 1), np.take_along_axis(fs, order, 1)
        iterations += 1
    return sim[:, 0], calls


def batch_refine(case, x, y, jobs, options=OPTIONS):
    """``warm_triples.refine`` for every (triple, best grid point) in ``jobs``, ``BLOCK`` at a time, with the columns
    loss. Each problem runs its own Nelder-Mead; blocks only bound the memory."""
    sign = (2 * y - 1).astype(np.float64)
    out = []
    for lo in range(0, len(jobs), BLOCK):
        block = jobs[lo:lo + BLOCK]
        problem = setup(case, x, y, block)
        best, calls = nelder_mead(lambda rows, z: losses(problem, rows, z, x, sign), len(block), **options)
        CALLS["calls"] += len(block)
        CALLS["evaluations"] += int(calls.sum())
        start, scale = problem[3], problem[4]
        out += [wt.score(case, x, y, t, start[k] + best[k] * scale[k]) for k, (t, _) in enumerate(block)]
    return out


def starts(case, px, py, triples, coarse):
    """The two start points #29's continuous fits refine from: the current values and the coarse optimum."""
    return [(t, wt.score(case, px, py, t, ct.current_values(case, t))) for t in triples] + list(zip(triples, coarse))


def committee(case, px, py, size=tc.COMMITTEE, pairs_kept=None, tie_break="coarse"):
    """``triple_committee.committee`` with its continuous fits refined together by ``batch_refine``."""
    if tie_break not in ("coarse", "largest-edit"):
        raise ValueError(f"unknown tie_break {tie_break!r}")
    pairs = tc.sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in tc.js.joint_order(tc.js.joint_scores(case, px, py, pairs, points=tc.ds.PAIR_POINTS))]
    triples = tc.t3.triples_from(ranked, tc.lr.entries(case["arrays"]), pairs_kept or tc.t3.PAIRS)
    coarse = [tc.t3.fit_triple(case, px, py, t, 7) for t in triples]
    acc7, cost7 = np.array([b[0][0] for b in coarse]), np.array([-b[0][1] for b in coarse])
    rank7 = np.empty(len(triples), int)
    rank7[tc.tr.order(acc7, cost7, None)] = np.arange(len(triples))
    refined = batch_refine(case, px, py, starts(case, px, py, triples, coarse))
    n = len(triples)
    fits = [ct.best_of(refined[k], refined[n + k]) for k in range(n)]
    accuracy, cost = np.array([b[0][0] for b in fits]), np.array([-b[0][1] for b in fits])
    if tie_break == "largest-edit":
        top = tc.tr.order(accuracy, cost, np.array([tc.largest_edit(case, b) for b in fits]))[:size]
    else:
        top = tc.tr.order(accuracy, cost, rank7)[:size]
    return [triples[k] for k in top], {triples[k]: fits[k] for k in top}, len(triples)


def install():
    tc.committee = committee


def run_seed(args):
    """#73's timed run of one case with columns everywhere and the committee's refines batched."""
    import cut_refine as cr
    import time_repair
    seed, out = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    cr.CALLS.update(calls=0, evaluations=0)
    CALLS.update(calls=0, evaluations=0)
    time_repair.run_seed((seed, out))
    row = json.loads(path.read_text())
    row.update(kind="batched", refine_calls=cr.CALLS["calls"] + CALLS["calls"],
               refine_evaluations=cr.CALLS["evaluations"] + CALLS["evaluations"],
               batched_calls=CALLS["calls"], batched_evaluations=CALLS["evaluations"])
    path.write_text(json.dumps(row) + "\n")
    return seed


def score(rows):
    """The batched run against #73 (``cut_refine.compare``) and against #74's columns rows on the same cases."""
    import cut_refine as cr
    before = {r["seed"]: r for r in json.loads((EVIDENCE / "time-repair" / "time-repair.json").read_text())}
    columns = {r["seed"]: r for r in json.loads((EVIDENCE / "cut-refine" / "columns.json").read_text())}
    out = cr.compare(rows, before)
    vs = cr.compare(rows, {s: columns[s] for s in rows})
    out["vs_columns"] = {k: vs[k] for k in ("core_hours_stage", "core_hours_stage_73", "core_hours_refine",
                                            "core_hours_refine_73", "speedup_stage", "same_pick", "same_winner",
                                            "same_disagreement", "crossed", "changed")}
    out["vs_columns"]["core_hours_stage_columns"] = out["vs_columns"].pop("core_hours_stage_73")
    out["vs_columns"]["core_hours_refine_columns"] = out["vs_columns"].pop("core_hours_refine_73")
    out["vs_columns"]["speedup_refine"] = vs["core_hours_refine_73"] / vs["core_hours_refine"]
    out["batched_share_of_refine_calls"] = sum(r["batched_calls"] for r in rows.values()) / out["refine_calls"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        rows = {json.loads(f.read_text())["seed"]: json.loads(f.read_text()) for f in a.score.glob("case-*.json")}
        out = EVIDENCE / "batched-refine"
        out.mkdir(parents=True, exist_ok=True)
        (out / "batched.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        result = score(rows)
        (out / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    import cut_refine as cr
    cr.install("columns")  # every refine outside the committee: the refits and the told triple
    import time_repair
    global batch_refine
    batch_refine = time_repair.CLOCK.tap(batch_refine, "margin refine")
    install()
    a.output.mkdir(parents=True, exist_ok=True)
    seeds = sorted(a.only, key=lambda s: -time_repair.seconds_65(s))
    with Pool(a.workers) as pool:  # forked after the swaps above
        for seed in pool.imap_unordered(run_seed, [(s, str(a.output)) for s in seeds]):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
