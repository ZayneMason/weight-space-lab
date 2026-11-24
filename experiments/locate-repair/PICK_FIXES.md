# Choose among the fixes from labels alone (roadmap #71)

**Date:** November 24, 2026 · **Status:** done · **Cost:** 20.2 core-hours

#70 found that three fixes each rescue a different one of the 6 cases #68 left over 0.01%, and none made any of those
cases worse, but choosing the best fix per case used the 1M-point probe, which a real repair cannot see. This
experiment runs the fixes on all 50 of #67's width-16 three-fault cases (seeds 250 to 299) and chooses among them
from labels only.

```bash
python experiments/locate-repair/pick_fixes.py --output runs/pick --workers 4
python experiments/locate-repair/pick_fixes.py --score runs/pick
```

For each case the script reruns #68 exactly (it reproduces #68's final pick in all 50) and builds up to four
candidates:

- **#68's pick.**
- **Restarts** (#70): the winner, and the true triple when it is in the committee, refitted from 20 starts.
- **Boundary labels** (#70): 4 rounds of 16 labels where the fitted network's margin is smallest, refitting the values.
- **Bigger committee** (#70): twice the committee and twice the pairs kept. #70 ran it where the committee lacked the
  true triple, which a repair cannot know; here it runs where #68's pick misses at least one of its own labels, a gate
  the repair can see. That gate fires on 5 cases (251, 258, 259, 261, 296) and includes all three of #70's.

Each candidate's network is its final one (the middle of #64's walk when it fits as many labels as the fit, else the
fit), exactly as #70 scored them. Two choosers then see labels no candidate was fitted on:

- **Uniform:** 64 fresh inputs uniform on [-2, 2]², labelled; most correct wins.
- **Disagree:** a fresh uniform pool of 200,000 inputs; up to 32 of those where any two candidates give different
  labels are labelled; most correct wins.

Ties keep #68's pick when it is among the best. The probe only scores. Results are in `evidence/pick-fixes/`.

## Results (measured)

| Choice | Under 0.01% | Under 0.1% | Exact triple | Worse than #68 |
|---|---|---|---|---|
| #68's pick | 44 | 48 | 37 | |
| Uniform chooser | 44 | 48 | 37 | none |
| **Disagree chooser** | **47** | **49** | **39** | 272 (0.0007% to 0.0013%) |
| Best by the probe (not available to a repair) | 47 | 49 | 39 | none |
| Restarts everywhere | 45 | 48 | 38 | none |
| Boundary labels everywhere | 45 | 48 | 37 | 253, 272, 286, 294 |
| Bigger committee where it ran | 46 | 49 | 39 | none |

1. **Labels where the candidates disagree choose as well as the probe does.** The disagree chooser puts 47 of 50
   under 0.01%, the same as choosing by the probe: it gains 258, 259 and 269 and loses none of #68's 44. It asks 341
   labels across the 50 cases, a median of 3; 7 cases have no input in the pool where the candidates differ, and
   only 5 use all 32.
2. **64 uniform labels cannot tell the candidates apart.** Every candidate gets all 64 right in all 50 cases, so the
   uniform chooser keeps #68's pick everywhere.
3. **In the three rescued cases the vote is lopsided.** 258: the bigger committee gets 31 of 32 against #68's 6.
   259: restarts 29 of 32 against 12. 269: boundary labels 30 of 32 against 2.
4. **The one case it makes worse stays far under the line.** In 272 it asks the only 2 split inputs, the boundary
   candidate gets both, and the probe moves from 0.0007% to 0.0013%. All 12 cases where the chooser and the probe
   disagree have every candidate under 0.01%, and the gap is under 0.001% in each.
5. **Boundary labels are not harmless everywhere.** #70 saw them help in all 6 of its cases; across all 50 they make 4
   cases slightly worse (253, 272, 286, 294), all still under 0.01%. Restarts and the bigger committee make none worse.
6. **The bigger committee's gate works.** Of the 5 cases where #68's pick misses a label it lifts 258 and 259 under
   the line; 261 and 296 still lack the true triple, and in 251 every candidate is already under 0.01%.
7. **Three cases stay over 0.01%:** 260 (0.044%), 261 (0.021%) and 296 (0.745%), the three #70 already explained:
   labels too few, a true triple that cannot fit them, and a true pair ranked 168th.

## Interpretation

- What #70 called an upper bound turns out to be reachable from labels: asking a handful of labels where the
  candidates disagree recovers the probe's choice in every case where the choice matters (measured on these 50). The
  probe is no longer needed to choose among the fixes.
- The cost is in building the candidates, not in choosing: 20.2 core-hours, against a median of 3 extra labels.
- The gate and the chooser were designed after seeing these 50 cases, so 47 of 50 is not yet a held-out result. #72
  runs the whole repair on fresh seeds.
