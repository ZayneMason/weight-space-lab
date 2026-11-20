"""Score rules for which stage's repair to keep, from saved per-stage results (roadmap #61).

The two-step rule always keeps the last stage it ran. #60 found that the 1,000-pair third stage helps at width 16
only when it reaches a perfect practice fit, and makes every width-32 case it reaches worse. This script rescores
each saved run under other final-choice rules without running anything: every rule only looks at practice
accuracy, which the repair sees, never at the million-input probe, which it does not. Writes
evidence/final-stage-choice/check.json.
"""
import json

from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "final-stage-choice"
ARMS = {"width_16": "two-step-unseen/two-step-unseen.json", "width_32": "width-32/width-32.json",
        "width_16_largest_edit": "largest-edit/largest-edit.json"}


def current(stages):
    return len(stages) - 1


def third_only_if_perfect(stages):
    """Keep the third stage only when it fits every practice label; otherwise the second."""
    if len(stages) < 3 or stages[2]["practice_accuracy"] == 1:
        return len(stages) - 1
    return 1


def best_practice_later(stages):
    """The stage with the highest practice accuracy, the later stage on a tie."""
    return max(range(len(stages)), key=lambda k: (stages[k]["practice_accuracy"], k))


def best_practice_earlier(stages):
    """The stage with the highest practice accuracy, the earlier (cheaper) stage on a tie."""
    return max(range(len(stages)), key=lambda k: (stages[k]["practice_accuracy"], -k))


def no_third_stage(stages):
    return min(len(stages) - 1, 1)


def oracle(stages):
    """Upper bound: the stage closest to the original on the probe, which no real rule can see."""
    return min(range(len(stages)), key=lambda k: stages[k]["disagreement"])


RULES = {f.__name__: f for f in (current, third_only_if_perfect, best_practice_later, best_practice_earlier,
                                 no_third_stage, oracle)}


def score(rows, rule):
    seeds = sorted(rows)
    pick = {s: rule(rows[s]["two_step"]) for s in seeds}
    end = {s: rows[s]["two_step"][pick[s]] for s in seeds}
    d = {s: end[s]["disagreement"] for s in seeds}
    # Only skipping the third stage saves CPU; every other rule still has to run it to see its practice fit.
    ran = lambda s: rows[s]["two_step"][:2] if rule is no_third_stage else rows[s]["two_step"]
    cpu = sum(st["seconds"] for s in seeds for st in ran(s))
    return {"under_0.1%": sum(v < 1e-3 for v in d.values()), "under_0.01%": sum(v < 1e-4 for v in d.values()),
            "exact": sum(e["faults_found"] == 3 for e in end.values()), "worst": max(d.values()),
            "cpu_hours": cpu / 3600,
            "changed": {str(s): [pick[s], d[s]] for s in seeds if pick[s] != len(rows[s]["two_step"]) - 1}}


def compare(arms):
    out = {}
    for name, rows in arms.items():
        res = {r: score(rows, f) for r, f in RULES.items()}
        base = {s: rows[s]["two_step"][-1]["disagreement"] for s in rows}
        for r, v in res.items():
            v["better"] = sorted(int(s) for s, (_, dd) in v["changed"].items() if dd < base[int(s)])
            v["worse"] = sorted(int(s) for s, (_, dd) in v["changed"].items() if dd > base[int(s)])
        out[name] = {"seeds": len(rows), "reached_third_stage": sorted(s for s in rows if len(rows[s]["two_step"]) == 3),
                     "rules": res}
    return out


def main():
    result = compare({name: cases(EVIDENCE / path) for name, path in ARMS.items()})
    OUT.mkdir(exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    for name, arm in result.items():
        print(name, arm["seeds"], "third stage:", arm["reached_third_stage"])
        for r, v in arm["rules"].items():
            print(f"  {r:24s} {v['under_0.1%']:3d} {v['under_0.01%']:3d} {v['exact']:3d} worst {v['worst']:.6f} "
                  f"cpu {v['cpu_hours']:.1f} better {v['better']} worse {v['worse']}")


if __name__ == "__main__":
    main()
