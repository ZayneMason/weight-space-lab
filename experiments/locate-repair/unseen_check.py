"""Score the two-step rule on 50 unseen three-fault cases, seeds 100 to 149 (roadmap #54).

Reads evidence/two-step-unseen/two-step-unseen.json (each case's finished two-step row, made by
``early_values.py --two-step``) and evidence/two-step-unseen/plain.json (each case's first-stage row, which is
the plain 200-pair, 200-triple run), and writes evidence/two-step-unseen/check.json.

``--plain-from DIR...`` rebuilds plain.json from the ``case-<seed>.stage0.json`` checkpoints in those run folders.
"""
import argparse
import json
from pathlib import Path

from wide_compare import EVIDENCE, cases, last, stats

LINES = (1e-3, 1e-4)
OUT = EVIDENCE / "two-step-unseen"


def score(final, plain):
    seeds = sorted(final)
    missing = [s for s in seeds if s not in plain]
    if missing:
        raise ValueError(f"no plain row for seeds {missing}")
    d = lambda arm, s: last(arm[s])["disagreement"]
    out = {"seeds": len(seeds), "two_step": stats(final, seeds), "plain": stats(plain, seeds)}
    for arm in ("two_step", "plain"):
        del out[arm]["median_seconds"]
    runs = {k: [s for s in seeds if final[s]["stages_run"] == k] for k in (1, 2, 3)}
    out["stages_run"] = {str(k): len(v) for k, v in runs.items()}
    out["reran_with_bigger_committee"] = sorted(runs[2] + runs[3])
    out["reran_with_wider_screen"] = runs[3]
    out["still_misses_a_label"] = [s for s in seeds if last(final[s])["practice_accuracy"] < 1]
    for line in LINES:
        out[f"line_{line}"] = {"gained": [s for s in seeds if d(final, s) < line <= d(plain, s)],
                               "lost": [s for s in seeds if d(plain, s) < line <= d(final, s)]}
    out["failures_left"] = [s for s in seeds if d(final, s) >= LINES[0]]
    out["above_0.01%_never_flagged"] = [s for s in runs[1] if d(final, s) >= LINES[1]]
    out["worst_seed"] = max(seeds, key=lambda s: d(final, s))
    out["cost_ratio"] = out["two_step"]["cpu_hours"] / out["plain"]["cpu_hours"]
    return out


def plain_from(dirs):
    rows = [json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).glob("case-*.stage0.json"))]
    return {"note": "first stage of each two-step run: 200 pairs, 200 triples, value labels in rounds 1 and 2",
            "cases": sorted(rows, key=lambda r: r["seed"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-from", type=Path, nargs="+")
    args = parser.parse_args()
    if args.plain_from:
        (OUT / "plain.json").write_text(json.dumps(plain_from(args.plain_from), indent=1) + "\n")
    result = score(cases(OUT / "two-step-unseen.json"), cases(OUT / "plain.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
