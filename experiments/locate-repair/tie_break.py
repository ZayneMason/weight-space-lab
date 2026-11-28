"""Ask labels where the duel's tied refits split (roadmap #91).

On #90's 50 fresh cases, 43 of #88's refit duels end with more than one candidate getting the most asked labels right,
because every refit is fitted on the labels it is then scored on. The true triple's refit is always among the tied,
and the winner is the pick if it is tied, otherwise the first candidate in the search's order; all 14 results short
of exact are such ties. The tied refits agree on the rest of #71's pool, so more rounds on it cannot split them.

This reruns #88's duel as it ran (same pool, draws and refits) and, where it ends tied, breaks the tie:

- draw a fresh pool of 200,000 inputs; each round, ask up to 64 where the tied networks split and that were not asked;
- score every tied network on the new labels before it is refitted on them (so a refit is never scored on labels it
  was fitted to), then refit the tied triples on #65's labels plus everything asked;
- keep only the networks with the highest score so far, and stop when one is left, when none split, or after 4 rounds;
- if more than one is left, the duel's rule decides among them (the pick if tied, else the search's order).

Then, as in #90, #82's flat-line step runs on the winner only where it is the pick. Fixed in advance, it runs on
#90's 50 fresh cases first and then on #89's 200.

    python experiments/locate-repair/tie_break.py --set fresh --output runs/b91f --workers 4
    python experiments/locate-repair/tie_break.py --set shaped --output runs/b91s --workers 4
    python experiments/locate-repair/tie_break.py --score runs/b91f runs/b91s
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import fresh_end as fe
import keep_one as k
import perfect_fit as pfit
from refit_duel import ASK, ROUNDS
from value_labels import line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "tie-break"
LINE = k.LINE
FRESH = 200_000


def fresh_cases():
    rows = {r["seed"]: r for r in json.loads((fe.OUT / "fresh-end.json").read_text())}
    out = {}
    for s, r in rows.items():
        pick = fe.pick_of(json.loads((fe.OUT / "first-part" / f"case-{s}.json").read_text()))
        out[s] = {"pick": pick, "triples": r["perfect"], "earlier": r}
    return out


def shaped_cases():
    picks = k.picks()
    rows = {r["seed"]: r for r in json.loads((EVIDENCE / "keep-two" / "keep-two.json").read_text())}
    return {s: {"pick": picks[s], "triples": r["perfect"], "earlier": r} for s, r in rows.items()}


def predict(case, nets, names, xs):
    return np.stack([k.ev.tf.predict(k.ev.tf.apply(case["arrays"], nets[n]), xs) for n in names])


def values(triple, edits):
    by = {(n, i): v for n, i, v in edits}
    return [by[tuple(w)] for w in triple]


def run_seed(args):
    seed, c, out = args
    path = out / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case, x, y = k.load(seed)
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    pick = c["pick"]
    triples = [tuple(tuple(w) for w in t) for t in c["triples"]]
    hits = lambda n: (pick["faults_found"] if n == "pick" else
                      len(k.ev.t3.hits(triples[int(n[1:])], case["faults"], case["width"])))
    nets = {"pick": [tuple(e) for e in pick["edits"]]}
    duel = tied = None
    rounds = []
    if triples:
        # #88's duel, exactly as refit_duel.run_seed runs it
        rng = np.random.default_rng([seed, 71])
        rng.uniform(-2, 2, size=(k.pf.UNIFORM, 2))
        pool = rng.uniform(-2, 2, size=(k.pf.POOL, 2))
        draw = np.random.default_rng([seed, 88])
        fit = lambda xs, ys, idx: {i: k.wt.fit(case, xs, ys, triples[i])[1] for i in idx}
        nets.update({f"t{i}": e for i, e in fit(x, y, range(len(triples))).items()})
        asked = np.zeros(0, dtype=int)
        for _ in range(ROUNDS):
            names = list(nets)
            votes = predict(case, nets, names, pool)
            split = np.setdiff1d(np.flatnonzero((votes != votes[0]).any(axis=0)), asked)
            if not len(split):
                break
            asked = np.concatenate([asked, draw.choice(split, size=min(ASK, len(split)), replace=False)])
            xs = np.concatenate([x, pool[asked]])
            ys = np.concatenate([y, k.ev.tf.predict(case["teacher"], pool[asked])])
            nets.update({f"t{i}": e for i, e in fit(xs, ys, range(len(triples))).items()})
        ya = k.ev.tf.predict(case["teacher"], pool[asked])
        correct = {n: int((k.ev.tf.predict(k.ev.tf.apply(case["arrays"], nets[n]), pool[asked]) == ya).sum())
                   for n in nets}
        best = max(correct.values())
        top = [n for n in nets if correct[n] == best]
        duel = {"winner": top[0], "asked": int(len(asked)), "tied": len(top),
                "true_tied": any(hits(n) == 3 for n in top)}
        alive = list(top)
        if len(top) > 1:
            # the tie-break, on a fresh pool, scoring each network before it is refitted
            frng = np.random.default_rng([seed, 91])
            fresh = frng.uniform(-2, 2, size=(FRESH, 2))
            fdraw = np.random.default_rng([seed, 91, 1])
            score = {n: 0 for n in top}
            got = np.zeros(0, dtype=int)
            for _ in range(ROUNDS):
                votes = predict(case, nets, alive, fresh)
                split = np.setdiff1d(np.flatnonzero((votes != votes[0]).any(axis=0)), got)
                if not len(split):
                    break
                new = fdraw.choice(split, size=min(ASK, len(split)), replace=False)
                yn = k.ev.tf.predict(case["teacher"], fresh[new])
                for n, v in zip(alive, predict(case, nets, alive, fresh[new])):
                    score[n] += int((v == yn).sum())
                got = np.concatenate([got, new])
                lead = max(score[n] for n in alive)
                alive = [n for n in alive if score[n] == lead]
                rounds.append({"split": int(len(split)), "asked": int(len(got)), "alive": len(alive),
                               "true_alive": any(hits(n) == 3 for n in alive)})
                if len(alive) == 1:
                    break
                xs = np.concatenate([x, pool[asked], fresh[got]])
                ys = np.concatenate([y, ya, k.ev.tf.predict(case["teacher"], fresh[got])])
                nets.update({f"t{i}": e for i, e in fit(xs, ys, [int(n[1:]) for n in alive if n != "pick"]).items()})
            tied = {"asked": int(len(got)), "rounds": rounds, "left": len(alive)}
    winner = alive[0] if triples else "pick"
    triple = tuple(tuple(w) for w in pick["winner"]) if winner == "pick" else triples[int(winner[1:])]
    dis = pick["disagreement"] if winner == "pick" else k.ev.aq.disagreement(case, nets[winner], probe)
    line = {"flat": False, "after": dis}
    if winner == "pick":
        arm = line_arm(case, x, y, triple, values(triple, nets["pick"]), probe, seed)
        line = {"flat": arm["flat"], "after": arm["rounds"][-1]["disagreement"] if arm["flat"] else dis}
    path.write_text(json.dumps({
        "seed": seed, "candidates": len(triples), "duel": duel, "tie_break": tied, "winner": winner,
        "chosen": {"triple": [list(w) for w in triple], "edits": k.ev.edit_list(nets[winner]),
                   "disagreement": dis, "faults_found": hits(winner)},
        "line": line, "result": [line["after"], hits(winner)],
        "seconds": round(time.perf_counter() - start, 1)}) + "\n")
    return seed


def earlier_result(r, shaped):
    """#90's result, or #89's network under #90's order (the line step only where the duel kept the pick)."""
    if not shaped:
        return tuple(r["result"])
    kept = r["duel"] is None or r["duel"]["winner"] == "pick"
    return (r["line"]["after"] if kept else r["chosen"]["disagreement"], r["chosen"]["faults_found"])


def score_set(folder, cases, shaped):
    rows = {r["seed"]: r for r in (json.loads(p.read_text()) for p in sorted(folder.glob("case-*.json")))}
    seeds = sorted(rows)
    before = {s: earlier_result(cases[s]["earlier"], shaped) for s in seeds}
    after = {s: tuple(rows[s]["result"]) for s in seeds}
    tied = [s for s in seeds if rows[s]["tie_break"]]
    return rows, {
        "seeds": seeds, "before": fe.tally(before), "after": fe.tally(after), "moves": fe.moves(seeds, before, after),
        "duel_reproduced": sorted(s for s in seeds if rows[s]["duel"] and cases[s]["earlier"]["duel"]
                                  and rows[s]["duel"]["winner"] == cases[s]["earlier"]["duel"]["winner"]),
        "tied": tied, "true_tied": sorted(s for s in tied if rows[s]["duel"]["true_tied"]),
        "settled": sorted(s for s in tied if rows[s]["tie_break"]["left"] == 1),
        "changed_winner": sorted(s for s in tied if rows[s]["winner"] != rows[s]["duel"]["winner"]),
        "labels_asked": sum(rows[s]["tie_break"]["asked"] for s in tied),
        "core_hours": sum(r["seconds"] for r in rows.values()) / 3600,
        "per_case": {str(s): {"before": before[s], "after": after[s], "winner": rows[s]["winner"],
                              "tied": rows[s]["duel"]["tied"] if rows[s]["duel"] else 0} for s in seeds}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", choices=("fresh", "shaped"))
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path, nargs=2, metavar=("FRESH", "SHAPED"))
    a = p.parse_args()
    k.SOURCES[700] = fe.OUT
    if a.score:
        OUT.mkdir(parents=True, exist_ok=True)
        check = {}
        for name, folder, cases, shaped in (("fresh", a.score[0], fresh_cases(), False),
                                            ("shaped", a.score[1], shaped_cases(), True)):
            rows, check[name] = score_set(folder, cases, shaped)
            (OUT / f"{name}.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(check, indent=1) + "\n")
        print(json.dumps({n: {key: v for key, v in c.items() if key not in ("per_case", "seeds")}
                          for n, c in check.items()}, indent=1))
        return
    k.cr.install("columns")
    k.br.install()
    pfit.install()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = fresh_cases() if a.set == "fresh" else shaped_cases()
    seeds = a.only or sorted(cases, key=lambda s: (-len(cases[s]["triples"]), s))  # the largest ties first
    with Pool(a.workers) as pool:  # forked after the installs above
        for s in pool.imap_unordered(run_seed, [(s, cases[s], a.output) for s in seeds]):
            print("done", s, time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
