"""Put the flat-line step on top of the search, on all 200 cases (roadmap #86).

After #85 the cases over the 0.01% line across #79's, #80's and #83's 200 cases are 331, 400 and 611, and none was
searched by #84, because their picks fit every one of #65's final labels. This runs #82's flat-line step
(``value_labels.line_arm``: the least-constrained direction of the margins, the middle of the segment of perfect fits
along it where it is flat, then 4 rounds of 8 labels along it) on every case's network after #85, from #65's final
labels:

- the chooser's network where #84's search ran (the pick, or the perfect fit it took);
- the chosen network elsewhere (#81's pick on #79's and #80's cases, the gated pick on #83's).

The result was fixed in advance as the network after the 32 labels where the direction is flat, and the network
before the step elsewhere, as in #83. Where the network is the same as #82's or #83's, the step's numbers should match
theirs exactly (the same labels and random draws), which the score checks.

    python experiments/locate-repair/flat_stack.py --output runs/f86 --workers 4
    python experiments/locate-repair/flat_stack.py --score runs/f86
    python experiments/locate-repair/flat_stack.py --truth   # the true broken weights' segment where a case stays over
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import keep_one as k
import value_labels as vl
from value_labels import line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "flat-stack"
SEARCHED = EVIDENCE / "perfect-fit" / "perfect-fit.json"
LINE = k.LINE


def chosen():
    """Every case's network after #85: its triple, values, disagreement, broken weights found and where it came from."""
    out = {}
    for s, p in k.picks().items():
        out[s] = {"from": "pick", "triple": [list(w) for w in p["winner"]], "edits": p["edits"],
                  "disagreement": p["disagreement"], "faults_found": p["faults_found"]}
    for r in json.loads(SEARCHED.read_text()):
        name = r["chooser"]["pick"]
        if name == "pick":
            continue
        q = r["perfect"][int(name[1:])]
        out[r["seed"]] = {"from": "search", "triple": q["triple"], "edits": q["edits"],
                          "disagreement": q["disagreement"], "faults_found": q["faults_found"]}
    return out


def values(net):
    by = {(n, i): v for n, i, v in net["edits"]}
    return [by[(n, i)] for n, i in net["triple"]]


def run_seed(args):
    seed, net, out = args
    path = out / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case, x, y = k.load(seed)
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    triple = tuple(tuple(w) for w in net["triple"])
    line = line_arm(case, x, y, triple, values(net), probe, seed)
    path.write_text(json.dumps({"seed": seed, "chosen": net, "labels_wrong": k.wrong(case, net["edits"], x, y),
                                "line": line, "seconds": round(time.perf_counter() - start, 1)}) + "\n")
    return seed


def result(row):
    line = row["line"]
    return line["rounds"][-1]["disagreement"] if line["flat"] else row["chosen"]["disagreement"]


def earlier():
    """#82's and #83's line steps by seed, with the network each ran on."""
    out = {}
    for r in json.loads((EVIDENCE / "value-labels" / "value-labels.json").read_text()):
        out[r["seed"]] = (r["winner"], r["line"])
    for r in json.loads((EVIDENCE / "fresh-line" / "fresh-line.json").read_text()):
        p = r["line_step"]["gated_pick"]
        out[r["seed"]] = (r["candidates"][p]["winner"], r["line_step"]["line"])
    return out


def score(folder):
    rows = {r["seed"]: r for r in (json.loads(p.read_text()) for p in sorted(folder.glob("case-*.json")))}
    seeds = sorted(rows)
    before = {s: rows[s]["chosen"]["disagreement"] for s in seeds}
    after = {s: result(rows[s]) for s in seeds}
    middle = {s: rows[s]["line"]["middle"]["disagreement"] if rows[s]["line"]["flat"] else before[s] for s in seeds}
    faults = {s: rows[s]["chosen"]["faults_found"] for s in seeds}  # the line step keeps the network's weights
    tally = lambda d: {"under_0.01%": sum(d[s] < LINE for s in seeds), "under_0.1%": sum(d[s] < 10 * LINE for s in seeds),
                       "exact": sum(faults[s] == 3 for s in seeds), "median": float(np.median([d[s] for s in seeds])),
                       "over_line": {str(s): d[s] for s in seeds if d[s] >= LINE}}
    flat = [s for s in seeds if rows[s]["line"]["flat"]]
    out = {"seeds": seeds, "before": tally(before), "middle": tally(middle), "after": tally(after),
           "flat": flat, "from_search": sorted(s for s in seeds if rows[s]["chosen"]["from"] == "search"),
           "closer": sorted(s for s in seeds if after[s] < before[s]),
           "further": sorted(s for s in seeds if after[s] > before[s]),
           "gained": sorted(s for s in seeds if after[s] < LINE <= before[s]),
           "lost": sorted(s for s in seeds if before[s] < LINE <= after[s]),
           "per_case_flat": {str(s): [before[s], middle[s], after[s]] for s in flat},
           "labels_wrong_before": {str(s): rows[s]["labels_wrong"] for s in seeds if rows[s]["labels_wrong"]}}
    old = earlier()
    same = [s for s in seeds if s in old and [list(w) for w in old[s][0]] == rows[s]["chosen"]["triple"]]
    agree = lambda s: (old[s][1]["flat"] == rows[s]["line"]["flat"] and
                       (not old[s][1]["flat"] or old[s][1]["rounds"][-1]["disagreement"] == after[s]))
    out["earlier_check"] = {"same_network": len(same), "agree": sum(agree(s) for s in same),
                            "disagree": [s for s in same if not agree(s)]}
    out["line_on_over"] = {str(s): {"flat": rows[s]["line"]["flat"], "singular_values": rows[s]["line"]["singular_values"],
                                    "segment": rows[s]["line"]["segment"], "before": before[s], "after": after[s],
                                    "faults_found": faults[s]} for s in seeds if before[s] >= LINE}
    out["core_hours"] = sum(r["seconds"] for r in rows.values()) / 3600
    return rows, out


def truth():
    """In each case still over the line: the true broken weights' least-constrained direction at their original values
    (which the repair cannot know), its segment of perfect fits, and the disagreement at its ends and middle."""
    check = json.loads((OUT / "check.json").read_text())
    rows = {r["seed"]: r for r in json.loads((OUT / "flat-stack.json").read_text())}
    out = {}
    for s in check["after"]["over_line"]:
        case, x, y = k.load(int(s))
        t = tuple(tuple(e) for e in case["faults"])
        scale = np.array([case["scale"][n] for n, _ in t])
        z = np.array([float(case["teacher"][n].flat[i]) for n, i in t]) / scale
        net = lambda v: [(n, i, float(w)) for (n, i), w in zip(t, v * scale)]
        probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
        d, a, b, sv = vl.segment(case, x, y, t, z, scale)
        out[s] = {"faults": case["faults"], "chosen_weights": rows[int(s)]["chosen"]["triple"],
                  "original_wrong": k.wrong(case, net(z), x, y),
                  "singular_values": sv, "segment": [float(a), float(b)],
                  "disagreement": {"start": k.ev.aq.disagreement(case, net(z + a * d), probe),
                                   "middle": k.ev.aq.disagreement(case, net(z + (a + b) / 2 * d), probe),
                                   "end": k.ev.aq.disagreement(case, net(z + b * d), probe)}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    p.add_argument("--truth", action="store_true")
    a = p.parse_args()
    if a.truth:
        result_ = truth()
        (OUT / "truth.json").write_text(json.dumps(result_, indent=1) + "\n")
        print(json.dumps(result_, indent=1))
        return
    if a.score:
        rows, result_ = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "flat-stack.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result_, indent=1) + "\n")
        print(json.dumps({k_: v for k_, v in result_.items() if k_ not in ("per_case_flat", "seeds")}, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    nets = chosen()
    seeds = a.only or sorted(nets)
    jobs = [(s, nets[s], a.output) for s in seeds]
    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(run_seed, jobs):
            print("done", s, time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
