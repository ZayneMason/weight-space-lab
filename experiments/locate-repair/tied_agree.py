"""Find out why the tied members agree (roadmap #69).

#68 ended with 41 of 50 cases (seeds 250 to 299) still having two or more committee members that fit every label,
and in 87 of 127 of its rounds no input in a pool of 20,000 split them. In 28 of the 41 the tie count is exactly
2^k for a winner with k output-layer edits, the mirror symmetry (#22): only the difference between the two output
logits matters, so an edit to one logit's weight or bias can be moved to the other's. This script reruns #68 on
the tied cases (same committees, refits, pools and labels), keeps every member tied at the end, and compares each
with the winner:

- ``mirror``: the member edits the winner's entries up to moving output-layer edits to the other logit.
- the entries it edits that the winner does not (mirrors folded), with the state of each entry's hidden unit on the
  usual 1M-point probe in the member's own network: ``dead`` (never active), ``linear`` (always active) or
  ``mixed``; output biases have no unit.
- each entry's edit size in its layer's scale, so an entry the fit barely moves shows as a free slot.
- how often its network and the winner's disagree on the probe.

    python experiments/locate-repair/tied_agree.py --output runs/agree --workers 4
    python experiments/locate-repair/tied_agree.py --score runs/agree
"""
import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from rebuild_committee import labels
from tied_labels import OUT as TIED, run_case, tied_top
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
OUT = EVIDENCE / "tied-agree"
OUTPUT_LAYER = ("weight", "bias")


def fold(entry, width):
    """An entry with any output-layer edit moved to logit 0, so an edit and its mirror fold to the same key."""
    name, index = entry[0], int(entry[1])
    if name == "weight":
        return (name, index % width)
    if name == "bias":
        return (name, 0)
    return (name, index)


def unit_of(entry, width):
    name, index = entry[0], int(entry[1])
    return {"hidden_weight": index // 2, "hidden_bias": index, "weight": index % width}.get(name)


def unit_states(arrays, x):
    """Per hidden unit on inputs ``x``: dead (never active), linear (always active) or mixed."""
    pre = x @ arrays["hidden_weight"].astype(np.float64).T + arrays["hidden_bias"].astype(np.float64)
    on = (pre > 0).mean(axis=0)
    return ["dead" if v == 0 else "linear" if v == 1 else "mixed" for v in on]


def size(case, entry, edits):
    """The entry's edit from the broken network's value, in its layer's scale (#57's scaled edit)."""
    name, index = entry[0], int(entry[1])
    value = next(v for n, i, v in edits if n == name and int(i) == index)
    return abs(float(value) - float(case["arrays"][name].flat[index])) / float(case["scale"][name])


def compare(case, winner, edits, member, member_edits, probe):
    """How ``member`` (with its edits) relates to the winner: mirror, extra entries and their units, disagreement."""
    width = case["width"]
    mine, theirs = {fold(e, width) for e in winner}, {fold(e, width) for e in member}
    arrays = ev.tf.apply(case["arrays"], member_edits)
    states = unit_states(arrays, probe[0])
    extra = []
    for e in member:
        if fold(e, width) in mine:
            continue
        u = unit_of(e, width)
        extra.append({"entry": [e[0], int(e[1])], "layer": e[0], "unit": u,
                      "state": states[u] if u is not None else "none", "size": size(case, e, member_edits)})
    ours = ev.tf.predict(ev.tf.apply(case["arrays"], edits), probe[0])
    return {"triple": [[e[0], int(e[1])] for e in member], "mirror": mine == theirs, "extra": extra,
            "edits": [[n, int(i), float(v)] for n, i, v in member_edits],
            "sizes": [size(case, e, member_edits) for e in member],
            "vs_winner": float((ev.tf.predict(arrays, probe[0]) != ours).mean()),
            "disagreement": float((ev.tf.predict(arrays, probe[0]) != probe[1]).mean()),
            "faults_found": len(ev.t3.hits(member, case["faults"], width))}


def run_seed(args):
    seed, out, stages = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.time()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    keep = {}
    row = run_case(case, x, y, ev.TWO_STEP[stages - 1], probe, seed, keep=keep)
    members, table = keep["members"], keep["table"]
    tied = tied_top(members, table)
    w = tied[0]  # most labels, then committee order: #68's pick
    states = unit_states(ev.tf.apply(case["arrays"], table[w][1]), probe[0])
    row.update(winner=[[e[0], int(e[1])] for e in w], winner_states=states,
               members=[compare(case, w, table[w][1], t, table[t][1], probe) for t in tied],
               stage_options=ev.TWO_STEP[stages - 1], seconds=round(time.time() - start, 1))
    path.write_text(json.dumps(row) + "\n")
    return seed


def fault_effects(seed):
    """Per fault, the probe disagreement of the original network with only that one fault put back (0 = silent)."""
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, labels_ = ev.bs.make_probe(case, ev.bs.PROBE)
    out = []
    for name, index, *_ in case["faults"]:
        broken = ev.tf.apply(case["teacher"], [(name, index, float(case["arrays"][name].flat[index]))])
        out.append(float((ev.tf.predict(broken, x) != labels_).mean()))
    return out


def mirrors_expected(winner, width):
    return 2 ** sum(e[0] in OUTPUT_LAYER for e in winner)


def check(rows, before, effects):
    """Per case, how many tied members are mirrors, agree with the winner on the probe, and what the others edit.

    ``effects`` maps each seed to its faults' one-at-a-time probe effects (``fault_effects``)."""
    per_case, kinds = {}, Counter()
    for s, r in rows.items():
        others = [m for m in r["members"] if not m["mirror"]]
        for m in others:
            kinds.update(f'{e["layer"]}:{e["state"]}' for e in m["extra"])
        per_case[str(s)] = {"tied": len(r["members"]), "mirrors": sum(m["mirror"] for m in r["members"]),
                            "mirrors_expected": mirrors_expected(r["winner"], 16),
                            "same_on_probe": sum(m["vs_winner"] == 0 for m in r["members"]),
                            "max_vs_winner": max(m["vs_winner"] for m in r["members"]),
                            "extra_states": sorted({e["state"] for m in others for e in m["extra"]}),
                            "extra_size_median": float(np.median([e["size"] for m in others for e in m["extra"]]))
                            if others else None,
                            "silent_faults": sum(v == 0 for v in effects[s]), "fault_effects": effects[s],
                            "matches_68": r["rounds"][-1]["winner"] == before[s]["rounds"][-1]["winner"]}
    pc = per_case.values()
    return {"seeds": len(rows), "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "matches_68": sum(v["matches_68"] for v in pc),
            "tied_members": sum(v["tied"] for v in pc), "mirror_members": sum(v["mirrors"] for v in pc),
            "same_on_probe_members": sum(v["same_on_probe"] for v in pc),
            "all_mirrors": sorted(int(s) for s, v in per_case.items() if v["mirrors"] == v["tied"]),
            "all_same_on_probe": sorted(int(s) for s, v in per_case.items() if v["same_on_probe"] == v["tied"]),
            "over_0.01%_from_winner": sorted(int(s) for s, v in per_case.items() if v["max_vs_winner"] >= 1e-4),
            "extra_entry_kinds": dict(sorted(kinds.items())),
            "with_silent_fault": sorted(int(s) for s, v in per_case.items() if v["silent_faults"]),
            "more_than_mirrors": sorted(int(s) for s, v in per_case.items() if v["tied"] > v["mirrors"]),
            "per_case": per_case}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    stages = cases(SOURCE / "region-labels.json")
    before = {r["seed"]: r for r in json.loads((TIED / "tied-labels.json").read_text())}
    tied = [s for s, r in before.items() if r["rounds"][-1]["tied"] >= 2]
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in tied}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tied-agree.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        with Pool(a.workers) as pool:
            effects = dict(zip(tied, pool.map(fault_effects, tied)))
        result = check(rows, before, effects)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(a.output), stages[s]["stages_run"]) for s in (a.only or tied)]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
