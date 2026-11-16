# Locate-and-repair: passing the test by fixing the wrong weight (roadmap #7, offline half)

**Date:** November 16, 2026 · **Status:** reference strategies done; model editors not run ·
**Cost:** CPU, about 5 minutes · **No model or API calls**

The editing benchmark names the entry to change, so it measures arithmetic and tool
discipline but not *finding* a fault. This analysis removes that hint and asks what
model-free strategies can do, as the network grows.

```bash
python experiments/locate-repair/analyze.py --widths 2 8 32 128 --seeds 100 --output runs/locate-repair
```

## Task

- A 2-input ReLU network with H ∈ {2, 8, 32, 128} hidden units and 2 outputs: 5H + 2
  parameters (12 to 642).
- A random teacher (redrawn if it predicts one class almost everywhere). One entry,
  chosen uniformly among all parameters, is shifted by ±1.6 times its tensor's standard
  deviation. Entries whose corruption is (nearly) invisible are redrawn: the task needs
  16 practice and 128 held-out inputs the corruption gets wrong.
- Practice and held-out sets, and the success rule, are the benchmark's own
  (`weight_space_lab.benchmark_tasks._dataset`): 16 + 16 practice examples; on 256
  held-out examples, at least 90% repair and zero regressions.
- The editor may change any one entry. It sees all weights and the practice inputs, and
  one no-change feedback call gives it all 32 practice labels.

## Strategies

| Strategy | What it does |
|---|---|
| given | Told the entry; takes its smallest edit that is perfect on practice (the best strategy in [task geometry](../task-geometry/README.md)) |
| min-change | Not told; for every entry finds its smallest practice-perfect edit, then picks the entry whose edit is smallest relative to its tensor's spread |
| any | Whether *some* entry's smallest practice-perfect edit succeeds (upper bound for rules that pick among them) |
| random / random-given | The benchmark's scripted random control: three uniform values in [−3, 3], chosen by practice accuracy, then fewer practice regressions; random entry, or the true one |

Values are searched on a 1e-3 grid within ±4 of the current value. "Damage undone"
compares the network with the teacher on 20,000 uniform inputs: 1 means it agrees
everywhere again, 0 means no better than before the edit.

## Results (measured, 100 cases per width)

| H | Parameters | given | min-change | any | random-given | random | min-change edited the corrupted entry | Entries with a practice-perfect edit (median) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 81% | 70% | 86% | 19.0% | 4.9% | 48% | 2 |
| 8 | 42 | 76% | 49% | 84% | 14.4% | 2.1% | 27% | 6 |
| 32 | 162 | 86% | 52% | 93% | 24.7% | 2.3% | 7% | 25 |
| 128 | 642 | 80% | 47% | 95% | 25.5% | 2.1% | **3%** | 104 |

![Success and localization against network size](figures/locate-repair.png)

1. **Hiding the location costs about 30 points, and the cost stops growing.** Against
   the location-given strategy, min-change loses 11 points at 12 parameters and 27–34
   points from 42 to 642 parameters, holding near 50%. The hypothesized collapse with
   size does not happen.
2. **Localization does collapse.** The min-change editor edits the corrupted entry in 48% of
   cases at H = 2 and 3% at H = 128. At H = 128, 44 of its 47 successes edit a *different*
   entry.
3. **Some single edit almost always passes.** The share of cases where at least one entry's
   smallest practice-perfect edit meets the rule grows from 86% to 95% with size; at
   H = 128 a median of 16 different entries pass.
4. **Passing the rule is not the same as undoing the damage.** Told the location, the
   smallest edit undoes a median 95–96% of the damage at every size. Not told, min-change
   undoes 95% at H = 2 but 75–79% at H ≥ 32. Among min-change *successes* at H = 32 and
   128, 15–19% undo less than half the damage.
5. **The random control stays near 2%** without the location, against 14–26% with it.

## Interpretation

- In bigger networks there are many single edits that restore behaviour on the inputs the
  benchmark samples, and practice data rarely singles out the real fault. An editor
  that only needs to pass the rule does not need to find the bug; an editor judged on
  finding it would fail almost always.
- A located benchmark should therefore score localization separately from passing, or
  test on dense inputs near the damaged region, or both.
- The corruption's footprint shrinks with width (median damage 14.9% of input space at
  H = 2, 1.5% at H = 128), which is part of why other entries can compensate. This is a
  property of the task generator, not only of the networks.

## Limits

- Model editors (the roadmap's main arms) were not run: they need paid API calls or CLI
  allowance and explicit approval.
- One corruption size (1.6 standard deviations) and one architecture family. The
  generator redraws invisible corruptions, so every case has a visible fault.
- Min-change is one reasonable selection rule among many; "any" bounds what a better rule
  could reach with practice data alone.

Evidence: [`evidence/locate-repair/locate-repair.json`](../../evidence/locate-repair/locate-repair.json)
(per-case rows and the summary). Code: `analyze.py`; 4 tests in `tests/test_locate_repair.py`.

Follow-up: [PRACTICE.md](PRACTICE.md) (roadmap #10) grows the practice set from 32 to
1,024 examples.
[SUBSTITUTES.md](SUBSTITUTES.md) (roadmap #14) asks which entries stand in for the
broken one: an exact output-logit mirror, or units that point the same way.
