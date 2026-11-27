"""Search 611 even though its pick fits every label (roadmap #87).

611 is the last of the 200 cases over the 0.01% line after #86. Its pick edits two of the three broken weights and a
wrong third, and fits all 1,096 of #65's final labels, so #84's search (gated on a missed label) never ran there. This
runs that search anyway, with #85's fitter: every triple that keeps one of the pick's weights, fitted on #65's final
labels, split over several processes; then #71's disagree chooser on the pick and every perfect fit, then #82's
flat-line step on the chooser's network. As a check not available to the repair, it also runs the line step on every
perfect fit and reports where each lands.

    python experiments/locate-repair/ungated_search.py --output runs/u87 --part 0 --parts 4   # parts 0 to 3
    python experiments/locate-repair/ungated_search.py --score runs/u87
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import keep_one as k
import perfect_fit as pfit
from value_labels import line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "ungated-search"
SEED = 611
LINE = k.LINE


def chosen(seed):
    """The case's network after #86's inputs: the pick (611 was not searched)."""
    return k.picks()[seed]


def run(out, part, parts, seed=SEED):
    k.cr.install("columns")
    k.br.install()
    pfit.install()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"case-{seed}.part{part}.json"
    if path.exists():
        return
    start = time.perf_counter()
    case, x, y = k.load(seed)
    pick = chosen(seed)
    all_triples = k.triples(case, pick["winner"])
    mine = all_triples[part::parts]
    found = []
    for t in mine:
        f = k.wt.fit(case, x, y, t)
        if f[0][0] == 1.0:
            found.append({"triple": [list(p) for p in t], "edits": k.ev.edit_list(f[1]),
                          "faults_found": len(k.ev.t3.hits(t, case["faults"], case["width"]))})
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    for p in found:
        p["disagreement"] = k.ev.aq.disagreement(case, [tuple(e) for e in p["edits"]], probe)
    path.write_text(json.dumps({"seed": seed, "part": part, "parts": parts, "searched": len(mine),
                                "total": len(all_triples), "perfect": found, "nudge_calls": dict(pfit.CALLS),
                                "seconds": round(time.perf_counter() - start, 1)}) + "\n")


def values(triple, edits):
    by = {(n, i): v for n, i, v in edits}
    return [by[(n, i)] for n, i in triple]


def score(folder, seed=SEED):
    parts = [json.loads(p.read_text()) for p in sorted(folder.glob(f"case-{seed}.part*.json"))]
    assert parts and len(parts) == parts[0]["parts"], "every part must be done"
    perfect = [p for q in parts for p in q["perfect"]]
    perfect.sort(key=lambda p: p["triple"])
    case, x, y = k.load(seed)
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    pick = chosen(seed)
    networks = {"pick": [tuple(e) for e in pick["edits"]],
                **{f"t{i}": [tuple(e) for e in p["edits"]] for i, p in enumerate(perfect)}}
    duel = k.chooser(case, networks, seed)
    name = duel["pick"]
    net = ({"triple": pick["winner"], "edits": pick["edits"], "disagreement": pick["disagreement"],
            "faults_found": pick["faults_found"]} if name == "pick" else perfect[int(name[1:])])
    triple = tuple(tuple(w) for w in net["triple"])
    line = line_arm(case, x, y, triple, values(triple, net["edits"]), probe, seed)
    after = line["rounds"][-1]["disagreement"] if line["flat"] else net["disagreement"]
    # Not available to the repair: the line step on every perfect fit.
    each = []
    for p in perfect:
        t = tuple(tuple(w) for w in p["triple"])
        scale = np.array([case["scale"][n] for n, _ in t])
        orig = np.array([float(case["teacher"][n].flat[i]) for n, i in t])
        ln = line_arm(case, x, y, t, values(t, p["edits"]), probe, seed)
        each.append({"triple": p["triple"], "faults_found": p["faults_found"], "disagreement": p["disagreement"],
                     "spreads_from_original": [round(float(d), 4) for d in
                                               (np.array(values(t, p["edits"])) - orig) / scale],
                     "flat": ln["flat"], "after_line": ln["rounds"][-1]["disagreement"] if ln["flat"] else None,
                     "middle": ln["middle"]["disagreement"] if ln["flat"] else None})
    out = {"seed": seed, "faults": case["faults"], "pick": {k_: pick[k_] for k_ in ("winner", "disagreement",
                                                                                     "faults_found")},
           "pick_wrong": k.wrong(case, pick["edits"], x, y), "labels": len(y),
           "searched": sum(q["searched"] for q in parts), "total": parts[0]["total"],
           "perfect_count": len(perfect), "true_perfect": any(p["faults_found"] == 3 for p in perfect),
           "chooser": duel, "chosen": {"name": name, "triple": net["triple"], "disagreement": net["disagreement"],
                                       "faults_found": net["faults_found"]},
           "line": {"flat": line["flat"], "singular_values": line["singular_values"], "segment": line["segment"],
                    "after": after},
           "result": {"disagreement": after, "faults_found": net["faults_found"], "under_line": after < LINE},
           "perfect": each,
           "nudge_calls": {key: sum(q["nudge_calls"][key] for q in parts) for key in parts[0]["nudge_calls"]},
           "core_hours": sum(q["seconds"] for q in parts) / 3600}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"case-{SEED}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k_: v for k_, v in result.items() if k_ != "perfect"}, indent=1))
        print(json.dumps(result["perfect"][:20]))
        return
    run(a.output, a.part, a.parts)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
