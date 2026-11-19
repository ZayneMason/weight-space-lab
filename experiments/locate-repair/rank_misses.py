"""Summarise where the true triple drops out for the nine fresh cases whose 1,000-pair committee lacks it (roadmap #51).

Reads evidence/rank-misses/rank-*.json, written by screen_ranks.py, and writes evidence/rank-misses/summary.json.
"""
import json

from wide_compare import EVIDENCE, cases, last

COMMITTEE = 200


def stage(rank):
    """Where the true triple is lost: never extended, or extended but cut from the committee."""
    if not rank["in_triples"]:
        return "never_extended"
    return "cut" if rank["rank_continuous"] >= COMMITTEE else "kept"


def summarise(ranks, fresh_narrow, fresh_wide):
    rows = []
    for r in sorted(ranks, key=lambda r: r["seed"]):
        s = r["seed"]
        rows.append({"seed": s, "best_true_pair_rank": r["true_subpair_ranks_joint"][0] + 1, "triples": r["triples"],
                     "stage": stage(r), "true_triple_rank": r.get("rank_continuous", -1) + 1,
                     "true_triple_practice_accuracy": r["told_acc"], "triples_tied_or_better": r["n_at_or_above_told"],
                     "member_200_practice_accuracy": r["acc_200th"],
                     "disagreement_200": last(fresh_narrow[s])["disagreement"], "disagreement_1000": last(fresh_wide[s])["disagreement"]})
    return {"cases": rows, "stages": {k: [r["seed"] for r in rows if r["stage"] == k] for k in ("never_extended", "cut", "kept")},
            "best_true_pair_rank_max": max(r["best_true_pair_rank"] for r in rows),
            "tied_or_better_range": [min(r["triples_tied_or_better"] for r in rows), max(r["triples_tied_or_better"] for r in rows)]}


def main():
    ranks = [json.loads(p.read_text()) for p in sorted((EVIDENCE / "rank-misses").glob("rank-*.json"))]
    out = summarise(ranks, cases(EVIDENCE / "adaptive-widen/adaptive-widen.json"), cases(EVIDENCE / "end-flag/end-flag.json"))
    (EVIDENCE / "rank-misses/summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out["cases"]:
        print(r)
    print(out["stages"], out["best_true_pair_rank_max"], out["tied_or_better_range"])


if __name__ == "__main__":
    main()
