"""Roadmap #57: record every triple's fit features so other tie-breaks among perfect fits can be scored.

Rebuilds one case's committee from the practice labels, exactly as screen_ranks.py does, and saves per triple:
continuous-fit accuracy and total edit, coarse-fit accuracy, total edit and rank (the current tie-break),
position in the screen's order, the largest single scaled edit, the smallest and mean practice margin of
the fitted repair, how many of its entries are biases, and whether it is the true triple.

    python experiments/locate-repair/tie_breaks.py 149 runs/tie-breaks 1000
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("early_values", HERE / "early_values.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["early_values"] = ev
spec.loader.exec_module(ev)
st, tc, t3, lr = ev.st, ev.tc, ev.t3, ev.lr
sc, js, ds, ct, tr = tc.sc, tc.js, tc.ds, tc.ct, tc.tr
wt = ct.wt


def features(seed, limit):
    case = ev.rs.resize(t3.corrupt(st.WIDTH, seed))
    px, py, _ = lr.split(case["practice"])
    truth = tuple(sorted(tuple(f[:2]) for f in case["faults"]))
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs, points=ds.PAIR_POINTS))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]), limit)
    coarse = [t3.fit_triple(case, px, py, t, 7) for t in triples]
    acc7, cost7 = np.array([b[0][0] for b in coarse]), np.array([-b[0][1] for b in coarse])
    rank7 = np.empty(len(triples), int)
    rank7[tr.order(acc7, cost7, None)] = np.arange(len(triples))
    fits = [ct.best_of(*ct.continuous_fits(case, px, py, t, c)) for t, c in zip(triples, coarse)]
    acc, cost = np.array([b[0][0] for b in fits]), np.array([-b[0][1] for b in fits])
    largest, low, mean = np.full(len(triples), np.nan), np.full(len(triples), np.nan), np.full(len(triples), np.nan)
    for k, (t, b) in enumerate(zip(triples, fits)):
        edits = b[1]
        largest[k] = max(abs(v - float(case["arrays"][n].flat[i])) / case["scale"][n] for n, i, v in edits)
        if acc[k] == 1:  # margins only matter among perfect fits
            m = wt.signed_margins(case, t, [v for _, _, v in edits], px, py) / case["scale"]["weight"]
            low[k], mean[k] = float(m.min()), float(m.mean())
    biases = np.array([sum(n.endswith("bias") for n, _ in t) for t in triples])
    is_true = np.array([tuple(sorted(t)) == truth for t in triples])
    return dict(seed=seed, limit=limit, acc=acc, cost=cost, acc7=acc7, cost7=cost7, rank7=rank7,
                screen=np.arange(len(triples)), largest=largest, margin_min=low, margin_mean=mean,
                biases=biases, is_true=is_true)


def main():
    seed, out = int(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else t3.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    f = features(seed, limit)
    np.savez_compressed(out / f"ties-{seed}-{limit}.npz", **f)
    print(seed, limit, len(f["acc"]), int((f["acc"] == 1).sum()), bool(f["is_true"].any()), flush=True)


if __name__ == "__main__":
    main()
