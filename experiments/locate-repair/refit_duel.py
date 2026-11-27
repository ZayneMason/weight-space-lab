"""Refit the candidates on the labels where they split (roadmap #88).

On 611 the pick and the true triple's fit both fit all 1,096 of #65's final labels, disagree on 67 of #71's 20,000
pool inputs, tie at 20 of the 32 labels #71's chooser asks there, and are both over the 0.01% line. The chooser only
scores the networks it is given. This duel moves them as it asks:

- the candidates are the pick (kept as it is, since it was fitted on labels this script does not have) and every
  triple #84's search found to fit all of #65's final labels, each fitted with #85's fitter;
- each round asks up to 64 of the pool inputs where the candidates' networks split and have not been asked yet (from
  #71's pool and draw), adds them to the labels, and refits every searched triple on #65's labels plus all asked;
- it stops when the candidates agree on every unasked pool input, or after 4 rounds;
- the winner gets the most asked labels right; a tie keeps the pick.

It runs on 611 (#87's two perfect fits) and on the 27 cases #84 searched (#85's perfect fits), the ones where the
search had candidates to duel.

    python experiments/locate-repair/refit_duel.py --output runs/r88 --workers 4
    python experiments/locate-repair/refit_duel.py --score runs/r88
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import keep_one as k
import perfect_fit as pfit
from wide_compare import EVIDENCE

OUT = EVIDENCE / "refit-duel"
LINE = k.LINE
ROUNDS, ASK = 4, 64


def candidates():
    """Every case with a search: the pick and the searched triples that fit all of #65's final labels."""
    picks = k.picks()
    out = {}
    for r in json.loads((EVIDENCE / "perfect-fit" / "perfect-fit.json").read_text()):
        out[r["seed"]] = {"pick": picks[r["seed"]], "triples": [p["triple"] for p in r["perfect"]],
                          "before": k.result_of(r, r["chooser"]["pick"])}
    u = json.loads((EVIDENCE / "ungated-search" / "case-611.json").read_text())
    out[611] = {"pick": picks[611], "triples": [p["triple"] for p in u["perfect"]],
                "before": [u["result"]["disagreement"], u["result"]["faults_found"]]}
    return out


def run_seed(args):
    seed, c, out = args
    path = out / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case, x, y = k.load(seed)
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    rng = np.random.default_rng([seed, 71])
    rng.uniform(-2, 2, size=(k.pf.UNIFORM, 2))  # #71's uniform labels, drawn as #71 draws them and not used here
    pool = rng.uniform(-2, 2, size=(k.pf.POOL, 2))
    draw = np.random.default_rng([seed, 88])
    triples = [tuple(tuple(w) for w in t) for t in c["triples"]]
    fit = lambda xs, ys: [k.wt.fit(case, xs, ys, t)[1] for t in triples]
    nets = {"pick": [tuple(e) for e in c["pick"]["edits"]]}
    nets.update({f"t{i}": e for i, e in enumerate(fit(x, y))})
    asked = np.zeros(0, dtype=int)
    rounds = []
    for _ in range(ROUNDS):
        names = list(nets)
        votes = np.stack([k.ev.tf.predict(k.ev.tf.apply(case["arrays"], nets[n]), pool) for n in names])
        split = np.setdiff1d(np.flatnonzero((votes != votes[0]).any(axis=0)), asked)
        if not len(split):
            break
        new = draw.choice(split, size=min(ASK, len(split)), replace=False)
        asked = np.concatenate([asked, new])
        xs = np.concatenate([x, pool[asked]])
        ys = np.concatenate([y, k.ev.tf.predict(case["teacher"], pool[asked])])
        nets.update({f"t{i}": e for i, e in enumerate(fit(xs, ys))})
        rounds.append({"split": int(len(split)), "asked": int(len(asked)),
                       "disagreement": {n: k.ev.aq.disagreement(case, nets[n], probe) for n in nets}})
    ya = k.ev.tf.predict(case["teacher"], pool[asked]) if len(asked) else np.zeros(0)
    correct = {n: int((k.ev.tf.predict(k.ev.tf.apply(case["arrays"], nets[n]), pool[asked]) == ya).sum())
               for n in nets}
    best = max(correct.values())
    winner = next(n for n in nets if correct[n] == best)
    found = lambda n: (c["pick"]["faults_found"] if n == "pick" else
                       len(k.ev.t3.hits(triples[int(n[1:])], case["faults"], case["width"])))
    dis = (c["pick"]["disagreement"] if winner == "pick" else k.ev.aq.disagreement(case, nets[winner], probe))
    path.write_text(json.dumps({
        "seed": seed, "candidates": len(nets), "rounds": rounds, "asked": int(len(asked)), "correct": correct,
        "winner": winner, "result": [dis, found(winner)], "before": c["before"],
        "true_index": next((f"t{i}" for i, t in enumerate(triples) if found(f"t{i}") == 3), None),
        "edits": k.ev.edit_list(nets[winner]), "seconds": round(time.perf_counter() - start, 1)}) + "\n")
    return seed


def score(folder):
    rows = {r["seed"]: r for r in (json.loads(p.read_text()) for p in sorted(folder.glob("case-*.json")))}
    seeds = sorted(rows)
    before = {s: tuple(rows[s]["before"]) for s in seeds}
    after = {s: tuple(rows[s]["result"]) for s in seeds}
    tally = lambda d: {"under_0.01%": sum(v[0] < LINE for v in d.values()), "exact": sum(v[1] == 3 for v in d.values()),
                       "over_line": {str(s): v[0] for s, v in d.items() if v[0] >= LINE}}
    out = {"seeds": seeds, "before": tally(before), "after": tally(after),
           "closer": sorted(s for s in seeds if after[s][0] < before[s][0]),
           "further": sorted(s for s in seeds if after[s][0] > before[s][0]),
           "gained": sorted(s for s in seeds if after[s][0] < LINE <= before[s][0]),
           "lost": sorted(s for s in seeds if before[s][0] < LINE <= after[s][0]),
           "took_other": sorted(s for s in seeds if rows[s]["winner"] != "pick"),
           "took_true": sorted(s for s in seeds if rows[s]["winner"] == rows[s]["true_index"]),
           "per_case": {str(s): {"before": before[s], "after": after[s], "winner": rows[s]["winner"],
                                 "asked": rows[s]["asked"], "rounds": len(rows[s]["rounds"]),
                                 "candidates": rows[s]["candidates"]} for s in seeds},
           "labels_asked": sum(r["asked"] for r in rows.values()),
           "core_hours": sum(r["seconds"] for r in rows.values()) / 3600}
    return rows, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        rows, result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "refit-duel.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k_: v for k_, v in result.items() if k_ not in ("per_case", "seeds")}, indent=1))
        return
    k.cr.install("columns")
    k.br.install()
    pfit.install()
    a.output.mkdir(parents=True, exist_ok=True)
    c = candidates()
    seeds = a.only or sorted(c, key=lambda s: (s != 611, s))
    with Pool(a.workers) as pool:  # forked after the installs above
        for s in pool.imap_unordered(run_seed, [(s, c[s], a.output) for s in seeds]):
            print("done", s, time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
