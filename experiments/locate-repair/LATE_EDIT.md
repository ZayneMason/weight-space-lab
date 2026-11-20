# The largest-edit tie-break only in the widest rerun (roadmap #59)

**Date:** November 20, 2026 · **Status:** done · **Cost:** CPU, about 9 core-hours (two and a half hours on four cores)

#58 found that ordering perfect practice fits by their largest single edit hurts when it picks the first
committee. #57's five wins were all measured at 1,000 pairs, the last stage of the two-step rule. This
experiment keeps the coarse order for the first two stages and uses the largest-edit order only in the
1,000-pair rerun.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --tie-break largest-edit --tie-break-from 2 \
    --only 52 55 68 102 --output runs/late-edit/h0                  # four groups side by side
python experiments/locate-repair/early_values.py --early 2 --two-step --tie-break largest-edit --tie-break-from 2 \
    --merge runs/late-edit/h*/early-values.json --output runs/late-edit/merged
python experiments/locate-repair/late_edit_check.py
```

`two_step(..., tie_break_from=k)` uses the requested tie-break from stage `k` on and the coarse order before it.
The run covers #51's nine lost cases (seeds 52, 55, 62, 63, 66, 68, 76, 93, 96) and the six unseen cases that
reach the third stage under #54 (102, 109, 124, 126, 129, 149). The six reuse #54's saved first two stages.
The comparison arm is the coarse-order two-step rule: #53's arm for seeds 52 to 96 and #54's run for the
unseen seeds. Results are in `evidence/late-edit/`.

## Results (measured)

| 15 cases | Coarse order | Largest-edit in the last stage |
|---|---:|---:|
| Reach the 1,000-pair stage | 6 | 6 |
| Under 0.1% disagreement | 15 | 15 |
| Under 0.01% | 7 | 8 |
| Exact three weights | 10 | 10 |
| True triple in the final committee | 11 | 11 |
| Last-stage CPU hours (six cases) | 4.34 | 4.53 |

1. **#51's nine cases never reach the stage where the new order applies.** Under the two-step rule all nine
   finish at the second stage, the 1,000-triple rerun, so their results are identical by construction. All of
   #57's five (55, 63, 76, 93, 96) already end on the exact three weights there with the coarse order. #57
   measured them at 1,000 pairs and 200 triples, a setting the two-step rule does not use for them.
2. **On the six that reach the last stage the change is small and mixed.** Four end closer (102, 126, 129,
   149) and two further (109, 124). Seed 102 crosses under 0.01% (0.0117% to 0.0070%). No case crosses the
   0.1% line in either direction.
3. **No case gains a true weight, and one loses one.** Seed 124 goes from one true weight at 0.012% to none at
   0.068%. The true triple is in the final committee in the same three of the six (102, 109, 129) under both
   orders, and 102, 109 and 129 stay exact.
4. **The three hard cases are still lost.** 124, 126 and 149 end without the true triple under both orders.
   The new order picks a different winning triple in 124, 126, 129 and 149.
5. **Cost** is about the same: the last stage takes 4% longer.

## Interpretation

- **The coarse order stays the default, and this line of tie-breaks is closed.** Applied early (#58) the
  largest-edit order makes things worse. Applied late it has almost nothing to work on, because the cases
  #57 found it could save never reach the late stage.
- **#57's measurement asked a question the full rule does not ask (inferred).** It rebuilt committees at
  1,000 pairs and 200 triples. Under the two-step rule those cases get 1,000 triples first, which already keeps
  the true triple in eight of nine. The remaining hard cases (124, 126) are lost under every tie-break #57
  tried, so no reordering of this committee will reach them.
- **What is left is a few hundredths of a percent.** Every one of these 15 repairs is under 0.1%
  disagreement. The remaining gap is about exactness, not behaviour, so the next question is whether the
  rule holds on a different network, not how to squeeze these cases further. That is #60.
