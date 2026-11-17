"""Roadmap #43: committee splits until the behaviour line, then #40's half mix, for three broken weights.

#42 switched all 8 labels per round to value queries once fewer than 20 of 20,000 pool inputs split the
committee. Under 0.01% fell from 16 to 13 of 25, and the true triple refitted on those labels did worse
too, so the last splitting inputs still carry information about the values. #40's fixed mix (4 value
queries and 4 committee splits every round, from the first) has the best behaviour so far (24 and 19 under
0.1% and 0.01%) but finds every visible fault in 17 cases against #37's 19.

This rule keeps all 8 labels on committee splits while at least LINE of the pool splits the committee,
exactly as #42, and below it uses #40's mix: 4 value queries, 4 of the most even splits. If #40's
visible-fault loss comes from value queries asked before the committee has found the weights, this should
keep the 19 and most of #40's behaviour gain.

    python experiments/locate-repair/half_line.py --output runs/half-line
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("switch_line", HERE / "switch_line.py")
sl = importlib.util.module_from_spec(_spec)
sys.modules["switch_line"] = sl
_spec.loader.exec_module(sl)
st, ss, vq, bs, tc, rs, t3, lr, cr, aq = sl.st, sl.ss, sl.vq, sl.bs, sl.tc, sl.rs, sl.t3, sl.lr, sl.cr, sl.aq

LINE = sl.LINE
VALUE_SHARE = 0.5


def choose(case, fits, edits, pool, rng, line=LINE, per_round=None, value_share=VALUE_SHARE):
    """#42's trigger, but below the line only ``value_share`` of the labels become value queries."""
    per_round = per_round or aq.PER_ROUND
    split = ss.committee_split(case, fits, pool)
    order = np.lexsort((rng.random(len(pool)), -split))  # the same draws as #41 and #42, so rounds match until the line
    splitting = int((split > 0).sum())
    if splitting >= line * len(pool):
        return [int(i) for i in order[:per_round]], {"value_queries": 0, "splitting": splitting}
    value = vq.value_queries(case, edits, pool, rng, int(round(per_round * value_share)))
    taken = set(value)
    filler = [int(i) for i in order if int(i) not in taken][:per_round - len(value)]
    return value + filler, {"value_queries": len(value), "splitting": splitting}


def analyse(case):
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    members, fits, count = tc.committee(case, px, py)
    silent = cr.silent_faults(case)
    history = st.run(case, members, fits, silent, bs.make_probe(case, bs.PROBE), chooser=choose)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
            "silent_faults": silent, "triples": count,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
            "switch": history, "seconds": round(time.time() - start, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=st.WIDTH)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            saved = args.output / f"case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            base = t3.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            rows.append(analyse(case))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            last = rows[-1]["switch"]
            print(seed, last[-1]["faults_found"], round(last[-1]["disagreement"], 6),
                  [h.get("value_queries") for h in last[:-1]], flush=True)
    report = {"schema": "weight-space-half-line-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH], "line": LINE,
              "value_share": VALUE_SHARE, "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL,
              "summary": st.summarize(rows), "cases": rows}
    (args.output / "half-line.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"][-1]))


if __name__ == "__main__":
    main()
