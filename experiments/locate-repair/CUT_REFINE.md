# Cut the margin refine's cost and check the picks hold (roadmap #74)

**Date:** November 25, 2026 · **Status:** done for #71's stage; #65's stage is #75 · **Cost:** 28.2 core-hours
recorded for two 35-case reruns (15.3 and 12.0 of repair, 0.9 of probe scoring), against the roadmap's estimate of
10 to 25.

#73 found that one step, the margin refine (`warm_triples.refine`: Nelder-Mead on a hinge loss of the label margins,
`xatol` 1e-6, `fatol` 1e-9, up to 800 iterations), takes 87% of #71's stage. A one-case profile (seed 264) showed
why: the committee's continuous fits call it twice for each of about 8,900 candidate triples, each call spends a
median of 274 loss evaluations, and each evaluation copies the whole network and recomputes it in Python.

`cut_refine.py` gives three cheaper refines and swaps one in for `warm_triples.refine` everywhere the repair calls it:

- **exact:** the same search with a loss that skips the copy and reuses the hidden layer when no hidden weight is
  edited. It is bit-identical to the original call by call (checked in the tests) but only about 15% faster per call,
  so it was not run on the 35 cases.
- **columns:** the same search with a loss that sums every hidden unit no edit touches once and recomputes only the
  rest. It matches the original loss to 12 places, so the search can take a slightly different path. About 2x per
  call.
- **loose:** columns with looser stopping (`xatol` 1e-4, `fatol` 1e-7, at most 300 iterations). About 2.6x per call.

Each 35-case run is #73's runner on #73's cases from the same saved labels, scored against #73's rows: core-hours,
the disagree chooser's pick, the network it chose and its disagreement with the original on a million probe inputs.

```bash
python experiments/locate-repair/cut_refine.py --kind columns --output runs/cut-columns --workers 5 \
    --only 250 256 264 272 280 287 295 302 309 314 320 326 333 340 348 259 275 283 291 297 \
           307 330 344 346 251 258 263 265 270 279 281 296 315 328 338
python experiments/locate-repair/cut_refine.py --kind loose --output runs/cut-loose --workers 5 --only ...
python experiments/locate-repair/cut_refine.py --score runs/cut-columns runs/cut-loose
```

Evidence is in `evidence/cut-refine/`: `columns.json` and `loose.json` (each case's candidates, picks, seconds by part
and step, refine calls and evaluations) and `check.json` (the score against #73).

## Results (measured)

| #71's stage, 35 cases | #73 (original) | columns | loose |
|---|---|---|---|
| Stage core-hours | 21.5 | 15.3 | 12.0 |
| of which margin refine | 18.6 | 11.7 | 8.4 |
| Speedup on the stage (median per case) | 1 | 1.40x (1.40x) | 1.79x (1.75x) |
| Evaluations per refine call | | 321 | 233 |
| Under 0.01% / exact | 34 / 28 | 34 / 28 | 34 / 28 |
| Same pick as #73 | | 35 of 35 | 32 of 35 |
| Same chosen network's disagreement | | 35 of 35 | 28 of 35 |
| Labels asked | 246 | 246 | 218 |

1. **The faster loss changes nothing but the time.** columns makes the same pick, chooses a network with the same
   disagreement and asks the same 246 labels in all 35 cases. It cuts the stage by 1.40x (1.18x to 1.73x per case).
2. **Looser stopping cuts more and still holds the line.** loose cuts the stage by 1.79x (1.60x to 2.12x per case)
   and keeps 34 under 0.01% and 28 exact, with no case crossing the line either way. It picks a different fix in 3
   cases (264, 344, 348) and moves the chosen network's disagreement in 7, all by under 0.001 percentage points apart
   from 296, the case that was already over (0.745% to 0.751%). Seed 344's new pick edits only one of the three
   broken weights, against two before, and stays under the line at 0.0004% (from 0.0002%).
3. **The stage gains less than the refine.** Per call, columns is about 2x and loose about 2.6x, but the stage gains
   1.40x and 1.79x. The refine is still 76% (columns) and 70% (loose) of the stage, and columns' searches run longer
   (321 evaluations per call against #73's typical 274), which eats part of the per-call gain.
4. **Two ideas were set aside before running:** refining only the triples with the best coarse fit, and dropping one
   of the two continuous fits. Both change which triples the committee sees, so they risk the picks in a way the
   cheaper loss does not (inferred, not measured).

## Interpretation

- The safe cut is columns: same answers, 30% less time. loose takes 44% off and reaches the same line on these 35
  cases, but it does change a few picks, so it needs a fresh-case check before it replaces the original.
- The refine is still most of the stage, so the next step on cost is to make it cheaper again, for example by
  running Nelder-Mead on many triples at once in arrays instead of one Python loop per triple (not tried).
- #65's stage is 58% of the whole repair (30.0 of 51.5 core-hours in #73) and uses the same functions, so the same
  swap should cut it too (inferred from the code). That is #75: time #65's stage by step with the original, columns
  and loose refines on a few cases, and check its repairs hold.
