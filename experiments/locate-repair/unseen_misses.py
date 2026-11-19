"""Where the two-step rule still loses the true triple on unseen seeds 124, 126 and 149 (roadmap #55).

Reads evidence/unseen-misses/rank-{seed}.json (200 pairs) and rank-{seed}-1000.json (1,000 pairs), written by
screen_ranks.py, and writes evidence/unseen-misses/summary.json.
"""
import json

from wide_compare import EVIDENCE, cases, last

HERE = EVIDENCE / "unseen-misses"
SEEDS = (124, 126, 149)


def where(rank, committee):
    """Never extended to triples, extended but cut from a committee of this size, or kept."""
    if not rank["in_triples"]:
        return "never_extended"
    return "cut" if rank["rank_continuous"] >= committee else "kept"


def summarise(ranks, final):
    rows = []
    for s in sorted(ranks):
        narrow, wide = ranks[s][200], ranks[s][1000]
        row = {"seed": s, "best_true_pair_rank": narrow["true_subpair_ranks_joint"][0] + 1,
               "true_pair_ranks": [k + 1 for k in narrow["true_subpair_ranks_joint"]],
               "true_triple_practice_accuracy": narrow["told_acc"],
               "final_disagreement": last(final[s])["disagreement"]}
        for pairs, r in ((200, narrow), (1000, wide)):
            row[f"pairs_{pairs}"] = {"triples": r["triples"], "true_triple_rank": r["rank_continuous"] + 1 if r["in_triples"] else None,
                                     "tied_or_better": r["n_at_or_above_told"],
                                     "committee_200": where(r, 200), "committee_1000": where(r, 1000)}
        rows.append(row)
    steps = {"200 pairs, 200 triples": ("pairs_200", "committee_200"), "200 pairs, 1,000 triples": ("pairs_200", "committee_1000"),
             "1,000 pairs, 200 triples": ("pairs_1000", "committee_200"), "1,000 pairs, 1,000 triples": ("pairs_1000", "committee_1000")}
    kept = {name: [r["seed"] for r in rows if r[p][c] == "kept"] for name, (p, c) in steps.items()}
    return {"cases": rows, "kept_by_setting": kept}


def main():
    ranks = {s: {200: json.loads((HERE / f"rank-{s}.json").read_text()),
                 1000: json.loads((HERE / f"rank-{s}-1000.json").read_text())} for s in SEEDS}
    out = summarise(ranks, cases(EVIDENCE / "two-step-unseen/two-step-unseen.json"))
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out["cases"]:
        print(r)
    print(out["kept_by_setting"])


if __name__ == "__main__":
    main()
