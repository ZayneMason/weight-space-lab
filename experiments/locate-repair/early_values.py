"""Roadmap #44: #40's half mix early, committee splits late, for three broken weights.

#43 found that value queries asked only once the committee has nearly settled change nothing: 22 and 16 of
25 under 0.1% and 0.01%, #37's cases exactly. The two cases #40 fixed (seeds 17 and 20) reach that point
late or never, so #40's gain must come from value labels asked early. #40 is also ahead on every visible
fault early (14 against 12 after 32 labels) and falls behind only between 32 and 40 labels.

So reverse the order: for the first EARLY rounds use #40's mix (4 value queries and 4 of the committee's
most even splits), then all 8 labels on the most even splits.

    python experiments/locate-repair/early_values.py --output runs/early-values
    python experiments/locate-repair/early_values.py --early 2 --output runs/switch-point-2   # roadmap #45
    python experiments/locate-repair/early_values.py --early 2 --widen-to 1000 --seed-start 50 --seeds 50 --output runs/adaptive-widen   # #49
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("half_line", HERE / "half_line.py")
hl = importlib.util.module_from_spec(_spec)
sys.modules["half_line"] = hl
_spec.loader.exec_module(hl)
st, ss, vq, bs, tc, rs, t3, lr, cr, aq = hl.st, hl.ss, hl.vq, hl.bs, hl.tc, hl.rs, hl.t3, hl.lr, hl.cr, hl.aq

EARLY = 4  # rounds of the half mix (32 labels), then committee splits only
VALUE_SHARE = hl.VALUE_SHARE


def make_chooser(early=EARLY, value_share=VALUE_SHARE):
    """A chooser for one case: #43's mix with the line set so it always mixes for ``early`` rounds, then never."""
    calls = []

    def choose(case, fits, edits, pool, rng, per_round=None):
        line = float("inf") if len(calls) < early else 0.0
        calls.append(line)
        return hl.choose(case, fits, edits, pool, rng, line=line, per_round=per_round, value_share=value_share)

    return choose


def best_practice_accuracy(members, fits):
    """The committee's best practice accuracy before any targeted label (#48's widening flag)."""
    return max(fits[t][0][0] for t in members)


def analyse(case, early=EARLY, pairs_kept=None, widen_to=None, committee_size=None):
    """Run one case. With ``widen_to``, rebuild the committee from that many pairs when no member of the
    first committee fits every practice label (roadmap #49). ``committee_size`` keeps that many triples
    instead of 200 (roadmap #52)."""
    size = {"size": committee_size} if committee_size else {}
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    members, fits, count = tc.committee(case, px, py, pairs_kept=pairs_kept, **size)
    first_best = best_practice_accuracy(members, fits)
    widened = bool(widen_to) and first_best < 1
    if widened:
        members, fits, count = tc.committee(case, px, py, pairs_kept=widen_to, **size)
    silent = cr.silent_faults(case)
    history = st.run(case, members, fits, silent, bs.make_probe(case, bs.PROBE), chooser=make_chooser(early))
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
            "silent_faults": silent, "triples": count,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
            "first_best_practice_accuracy": first_best, "widened": widened, "committee_size": len(members),
            "switch": history, "seconds": round(time.time() - start, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=st.WIDTH)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--early", type=int, default=EARLY, help="rounds of the half mix before splits only (#45)")
    parser.add_argument("--pairs", type=int, default=None, help="pairs extended to triples (default 200; #47)")
    parser.add_argument("--widen-to", type=int, default=None,
                        help="rebuild the committee from this many pairs when none fits every practice label (#49)")
    parser.add_argument("--committee", type=int, default=None, help="triples kept in the committee (default 200; #52)")
    parser.add_argument("--only", type=int, nargs="+", help="run only these seeds (#47)")
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in args.only or range(args.seed_start, args.seed_start + args.seeds):
            saved = args.output / f"case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            base = t3.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            rows.append(analyse(case, args.early, args.pairs, args.widen_to, args.committee))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            last = rows[-1]["switch"]
            print(seed, last[-1]["faults_found"], round(last[-1]["disagreement"], 6),
                  [h.get("value_queries") for h in last[:-1]], flush=True)
    report = {"schema": "weight-space-early-values-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH], "early_rounds": args.early, "pairs_kept": args.pairs or t3.PAIRS, "widen_to": args.widen_to, "committee": args.committee or tc.COMMITTEE,
              "value_share": VALUE_SHARE, "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL,
              "summary": st.summarize(rows), "cases": rows}
    (args.output / "early-values.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"][-1]))


if __name__ == "__main__":
    main()
