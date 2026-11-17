# Switching to value queries once the committee agrees (roadmap #39)

**Date:** November 17, 2026 · **Status:** done for two faults (the three-fault half moved to roadmap
#40) · **Cost:** CPU, about an hour and a half (four runs in parallel, two arms per case)

[#38](VALUE_QUERIES.md)'s best arm, a refined pair fit with half of every round's labels spent on value
queries, brought 24 of 25 two-fault repairs under 0.1% disagreement, but found every visible fault in
22 of 25 against 23 without value queries. #37's logs suggested why a switch could keep both: within a
few rounds the committee (the pairs within 0.5% of the best practice accuracy) labels every pool input
the same way, so its "targeted" labels after that are effectively random.

```bash
python experiments/locate-repair/switch_queries.py --output runs/switch-queries
```

## Method

Same 25 cases, refined fit and million-input probe as #38. Two rules for each round's 8 labels:

- **switch:** if any of the 20,000 pool inputs splits the committee, all 8 labels go to the most even
  splits; if none does, all 8 are value queries.
- **fill:** targeted labels only on inputs that split the committee, most even first, and value
  queries for the rest of the 8.

## Results (measured)

**After all 64 labels, 25 cases** (#38's rows for comparison):

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | Under 0.1% | Under 0.01% | Median disagreement |
|---|---:|---:|---:|---:|---:|---:|
| #38 `refine` (no value queries) | 0 | 19 | 23 | 22 | 13 | 0.0089% |
| #38 `refine_value` (half) | 745 | 18 | 22 | 24 | 20 | 0.0016% |
| `switch` | 498 | 19 | **23** | **24** | 19 | 0.0011% |
| `fill` | 607 | 18 | 22 | 24 | 20 | 0.0019% |

**Round by round, `switch` against #38's `refine_value`:**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `switch`: under 0.1% | 10 | 17 | 19 | 23 | 24 | 24 | 24 | 24 | 24 |
| `refine_value`: under 0.1% | 10 | 19 | 21 | 23 | 24 | 24 | 24 | 24 | 24 |
| `switch`: under 0.01% | 1 | 1 | 4 | 8 | 10 | 14 | 18 | 19 | 19 |
| `refine_value`: under 0.01% | 1 | 3 | 9 | 17 | 18 | 20 | 22 | 22 | 20 |

1. **Switching keeps both columns.** `switch` finds every visible fault in 23 of 25, as without value
   queries, and still ends with 24 under 0.1% and 19 under 0.01%, level with #38's best arm. It spends
   498 labels on value queries, against 745 for `refine_value` (which asks for 4 a round and sometimes
   finds fewer inputs that its nudged copies disagree on).
2. **But it gets there later.** Under the 0.01% line, `switch` is at 8 of 25 after 24 labels, against
   17 for `refine_value`, and only catches up after 48. Early value queries are what make the strict
   line come early.
3. **Neither arm is closer to the original case by case.** Against `refine_value`, `switch` ends with
   lower disagreement in 7 cases and higher in 13; `fill` in 9 and 12. Against `refine`, `switch` is
   lower in 16 and higher in 5.
4. **The committee often re-splits.** In 7 of the 25 `switch` cases, a round with all value queries is
   followed by one with targeted labels, because the refit moves a pair back within the margin.
   Seeds 10, 14 and 20 never switch.
5. **`fill` buys nothing over a fixed half.** It ends at 22 every visible fault and 24 and 20 on
   behaviour, the same as `refine_value`.
6. **Seed 23 still fails** in both arms, on a wrong weight at 0.10%.
7. **Cost:** median 207 and 204 seconds per arm per case, against 198 for #38's refined arms.

## Interpretation

- *Inference:* the one-case loss in visible faults that #38 blamed on value queries is the size of
  the noise here. Switching recovers it, but the price is a slower approach to the strict line, so
  which rule to prefer depends on the label budget: a fixed half for small budgets, a switch when there
  are enough labels to finish both jobs.
- *Inference:* a committee that agrees on every pool input is not a stable signal, because the next
  refit can bring another pair back within the margin. A rule that waits for agreement over two rounds,
  or that compares the leader with the runner-up directly, might switch more cleanly. Not tested.
- One run per arm on 25 cases, so one or two cases either way is within noise.

## Evidence

- `evidence/switch-queries/switch-queries.json`: every case, both arms, every round.
