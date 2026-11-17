"""Roadmap #42: switch to value queries at the behaviour line, for three broken weights.

#41 ran #39's switch rule (committee splits while any pool input splits the committee, value queries
after) on three faults and it almost never switched: in 16 of 25 cases some pool input still split
the committee in every round, so only 101 of 1,600 labels were value queries and the result was #37's.
The committee keeps triples that behave alike almost everywhere, and a few pool inputs still tell them
apart.

This rule switches once fewer than LINE of the pool (0.1%, or 20 of 20,000 inputs) split the
committee, the same line the repairs are scored on. While at least that many split it, the 8 labels go
to the most even splits, exactly as in #41; below it, all 8 go to value queries. The number of
splitting inputs is recorded every round.

    python experiments/locate-repair/switch_line.py --output runs/switch-line
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("switch_triples", HERE / "switch_triples.py")
st = importlib.util.module_from_spec(_spec)
sys.modules["switch_triples"] = st
_spec.loader.exec_module(st)
ss, vq, bs, tc, rs, t3, lr, cr, aq = st.ss, st.vq, st.bs, st.tc, st.rs, st.t3, st.lr, st.cr, st.aq

LINE = 0.001


def choose(case, fits, edits, pool, rng, line=LINE, per_round=None):
    """#41's switch rule with the trigger moved: value queries once under ``line`` of the pool splits the committee."""
    per_round = per_round or aq.PER_ROUND
    split = ss.committee_split(case, fits, pool)
    order = np.lexsort((rng.random(len(pool)), -split))  # the same draws as ss.choose, so rounds match #41 until a switch
    splitting = int((split > 0).sum())
    if splitting >= line * len(pool):
        return [int(i) for i in order[:per_round]], {"value_queries": 0, "splitting": splitting}
    value = vq.value_queries(case, edits, pool, rng, per_round)
    filler = [int(i) for i in order if int(i) not in set(value)][:per_round - len(value)]
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
                  [h.get("splitting") for h in last[:-1]], flush=True)
    report = {"schema": "weight-space-switch-line-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH], "line": LINE,
              "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL, "summary": st.summarize(rows), "cases": rows}
    (args.output / "switch-line.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"][-1]))


if __name__ == "__main__":
    main()
