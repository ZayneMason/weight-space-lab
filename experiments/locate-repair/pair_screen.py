"""Cheap screens for pairs of broken weights (roadmap #23)

Roadmap #21's all-pairs search found the broken pair at H = 8 in about half the cases, and #22's
targeted queries raised that to 20 of 25, but fitting every pair (861 at H = 8, 13,041 at H = 32)
does not scale. This experiment ranks every pair with one of two cheap screens, then runs #21's
joint fits only on the best-ranked pairs.

**Newton** screen: the margin m(x) = logit_1 - logit_0 of the corrupted network, labels s = +-1 from the
practice set, loss softplus(-s m). With J the per-example gradient of m (N x P), the gradient
g = J^T r and the Gauss-Newton matrix G = J^T W J (+ a small ridge), a pair S scores
g_S^T G_SS^-1 g_S, the loss drop a Newton step on those two entries predicts.

**Coverage** screen: for every entry alone, the grid value (#7's grid) that corrects the most
repair examples without breaking a protected one; a pair scores the share of repair examples
that its two entries' corrected sets cover together (ties: higher single-edit accuracy). Two
faults in different units each cause their own errors, so each true entry alone should fix
most of its own.

Per case: the true pair's rank in the screen (any pair that counts both faults, mirrors included),
then #21's coarse 61 x 61 fit of the top K pairs, the 20 best re-fitted on the 200 x 200 grid.

    python experiments/locate-repair/pair_screen.py --widths 8 32 --seeds 50 --top 200 1000 --output runs/pair-screen
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pair_search", HERE / "pair_search.py")
ps = importlib.util.module_from_spec(_spec)
sys.modules["pair_search"] = ps
_spec.loader.exec_module(ps)
tf, lr = ps.tf, ps.lr

RIDGE = 1e-6


def margin_jacobian(arrays, x):
    """d(logit_1 - logit_0)/d(entry) per example, columns in ``lr.entries`` order, and the margin."""
    a = {n: v.astype(np.float64) for n, v in arrays.items()}
    z = x @ a["hidden_weight"].T + a["hidden_bias"]
    active = (z > 0).astype(np.float64)
    h = z * active
    diff = a["weight"][1] - a["weight"][0]  # (H,)
    margin = h @ diff + a["bias"][1] - a["bias"][0]
    upstream = active * diff  # (N, H): d margin / d z
    columns = [(upstream[:, :, None] * x[:, None, :]).reshape(len(x), -1),  # hidden_weight, index u*2+j
               upstream,                                                      # hidden_bias
               -h, h,                                                         # weight rows 0 and 1
               -np.ones((len(x), 1)), np.ones((len(x), 1))]                   # bias 0 and 1
    return np.concatenate(columns, axis=1), margin


def pair_scores(arrays, x, y):
    """Predicted loss drop for every pair (i < j), as a dense upper-triangular matrix."""
    jac, margin = margin_jacobian(arrays, x)
    s = 2.0 * y - 1.0
    p = 1 / (1 + np.exp(s * margin))  # sigma(-s m)
    g = jac.T @ (-s * p)
    gram = (jac * (p * (1 - p))[:, None]).T @ jac
    d = np.diag(gram) + RIDGE * (np.trace(gram) / len(gram) + 1e-12)
    off = gram
    det = d[:, None] * d[None, :] - off ** 2
    # a mirror pair's two columns are exact opposites: the ridge keeps det > 0, and the pair
    # scores what one of its entries scores alone (one direction, not two)
    det = np.where(det > 1e-18 * d[:, None] * d[None, :], det, np.inf)
    score = (d[None, :] * g[:, None] ** 2 + d[:, None] * g[None, :] ** 2 - 2 * off * g[:, None] * g[None, :]) / det
    return np.triu(score, 1), g ** 2 / d


def coverage_scores(arrays, x, y, repair_mask=None):
    """Share of wrong practice examples that each pair's two single edits correct together."""
    evaluator, entries = lr.Evaluator(arrays, x), lr.entries(arrays)
    right = tf.predict(arrays, x) == y
    fixed, best = [], []
    for name, index in entries:
        current = float(arrays[name].flat[index])
        grid = np.round(current + np.arange(-lr.SPAN, lr.SPAN + lr.STEP / 2, lr.STEP), 6)
        correct = evaluator.predict(name, index, grid) == y
        broken = (~correct & right).sum(axis=1)
        gained = (correct & ~right).sum(axis=1)
        k = int(np.argmax(np.where(broken == 0, gained, -1))) if (broken == 0).any() else int(np.argmin(broken))
        fixed.append(correct[k] & ~right)
        best.append(correct.mean(axis=1).max())
    fixed, best = np.array(fixed, dtype=np.float64), np.array(best)
    union = fixed.sum(1)[:, None] + fixed.sum(1)[None, :] - fixed @ fixed.T
    cover = union / max((~right).sum(), 1)
    return np.triu(cover + 1e-3 * (best[:, None] + best[None, :]), 1), cover


def order_pairs(arrays, score):
    i, j = np.triu_indices(len(score), 1)
    order = np.argsort(-score[i, j], kind="stable")
    entries = lr.entries(arrays)
    return [(entries[i[k]], entries[j[k]]) for k in order]


def ranked_pairs(arrays, x, y, screen="coverage"):
    score = pair_scores(arrays, x, y)[0] if screen == "newton" else coverage_scores(arrays, x, y)[0]
    return order_pairs(arrays, score)


def true_rank(pairs, faults, width):
    return next(k for k, p in enumerate(pairs) if ps.found(p, faults, width) == 2)


def analyse(case, tops):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    newton = ranked_pairs(case["arrays"], px, py, "newton")
    pairs = ranked_pairs(case["arrays"], px, py, "coverage")
    rank = true_rank(pairs, faults, width)
    row = {"width": width, "seed": case["seed"], "faults": faults, "pairs": len(pairs),
           "newton_rank": true_rank(newton, faults, width), "true_pair_rank": rank}
    for top in tops:
        fits = {pair: ps.pair_fit(case, px, py, pair) for pair in pairs[:top]}
        shortlist = sorted(fits, key=fits.get, reverse=True)[:ps.REFINE]
        fine = {pair: ps.fine_fit(case, px, py, pair) for pair in shortlist}
        winner = max(fine, key=lambda p: fine[p][0])
        (accuracy, _), edits = fine[winner]
        row[f"top{top}"] = dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                                screened_has_both=rank < top,
                                refined_has_both=any(ps.found(p, faults, width) == 2 for p in fine),
                                faults_found=ps.found(winner, faults, width), practice_accuracy=accuracy,
                                winner=[list(e) for e in winner])
    return row


def summarize(rows, tops):
    out = {}
    for width in sorted({r["width"] for r in rows}):
        group = [r for r in rows if r["width"] == width]
        entry = {"cases": len(group), "pairs": group[0]["pairs"]}
        for screen, key in (("newton", "newton_rank"), ("coverage", "true_pair_rank")):
            ranks = np.array([r[key] for r in group])
            entry[screen] = {"true_pair_rank_median": float(np.median(ranks)),
                             "true_pair_in_top": {str(k): float(np.mean(ranks < k)) for k in (1, 10, 50, 200, 1000)}}
        for top in tops:
            rs = [r[f"top{top}"] for r in group]
            entry[f"top{top}"] = {"success": float(np.mean([r["success"] for r in rs])),
                                  "found_both": float(np.mean([r["faults_found"] == 2 for r in rs])),
                                  "found_one": float(np.mean([r["faults_found"] == 1 for r in rs])),
                                  "screened_has_both": float(np.mean([r["screened_has_both"] for r in rs])),
                                  "refined_has_both": float(np.mean([r["refined_has_both"] for r in rs])),
                                  "damage_undone_median": float(np.median([r["damage_undone"] for r in rs]))}
        out[str(width)] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--top", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--merge", type=Path, nargs="+", help="merge pair-screen.json files into --output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: (r["width"], r["seed"]))
    else:
        rows = []
        for width in args.widths:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                case = tf.corrupt(width, seed)
                if case is not None:
                    rows.append(analyse(case, args.top))
                    print(width, seed, rows[-1]["newton_rank"], rows[-1]["true_pair_rank"], flush=True)
    report = {"schema": "weight-space-pair-screen-v1", "tops": args.top, "summary": summarize(rows, args.top), "cases": rows}
    (args.output / "pair-screen.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
