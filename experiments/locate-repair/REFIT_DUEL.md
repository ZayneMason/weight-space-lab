# Refit the candidates on the labels where they split (roadmap #88)

**Date:** November 27, 2026 · **Status:** done on 611 and the 27 cases #84 searched; a cheap search that runs on
every case, so that a case like 611 is searched without being picked by hand, is #89 · **Cost:** 0.52 core-hours
recorded, against the roadmap's estimate of a few core-minutes (645 alone has 413 candidates).

On 611 the pick and the true triple's fit both fit all 1,096 of #65's final labels and tie on #71's chooser's 32
labels, and both are over the 0.01% line. #71's chooser only scores the networks it is given. `refit_duel.py` moves
them as it asks:

- the candidates are the pick, kept as it is (it was fitted on labels this script does not have), and every triple
  #84's search found to fit all of #65's final labels, each fitted with #85's fitter;
- each round asks up to 64 of the pool inputs (#71's pool) where the candidates' networks split and that were not
  asked yet, adds them to the labels, and refits every searched triple on #65's labels plus all asked;
- it stops when the candidates agree on every unasked pool input, or after 4 rounds;
- the winner gets the most asked labels right, and a tie keeps the pick.

It runs on 611 (#87's two perfect fits) and on the 27 cases #84 searched (#85's perfect fits).

```bash
python experiments/locate-repair/refit_duel.py --output runs/r88 --workers 4
python experiments/locate-repair/refit_duel.py --score runs/r88
```

Evidence is in `evidence/refit-duel/`: `refit-duel.json` (every case's rounds, each candidate's disagreement after
each round, the labels each got right, the winner's edits and seconds) and `check.json` (the score).

## Results (measured)

| 28 cases (611 and #84's 27) | under 0.01% | exact | over the line |
|---|---|---|---|
| #71's chooser on the same candidates (#85, #87) | 27 | 22 | 611 (0.0118%) |
| Refit duel (the result) | **28** | **24** | none |

| All 200 cases | under 0.01% | exact | over the line |
|---|---|---|---|
| After #86 | 199 | 158 | 611 |
| With the refit duel on these 28 (611's search run by hand) | **200** | **160** | none |

1. **611 is fixed exactly.** In the first round the duel asks 64 of the 67 split inputs and refits both searched
   triples: the true triple's fit goes from 0.0160% to 0.0032% and the pick's refitted triple from 0.0282% to
   0.0227%. After a second round of 52 the true triple is at 0.0008% and the networks agree on the rest of the pool.
   On the 116 labels asked, the true triple gets 116 right, the pick 84 and its refitted triple 91.
2. **Refitting makes the true triple win.** The duel takes a searched triple in 20 of 28 cases, and it is the true
   broken weights in 18. It makes 13 cases closer and 6 further, all under the line. The largest gains: 648 from
   0.0037% to 0.0000%, 443 from 0.0033% to 0.0005%, 547 from 0.0058% to 0.0016%, 528 from 0.0057% to 0.0019%.
3. **Two cases become exact and one stops being.** 524 (0.0010% to 0.0005%) and 528 move from two broken weights to
   three, and 611 from two to three. 522 moves from three to two (0.0010% to 0.0025%), and 419 from one to two (0.0032%
   to 0.0039%). The other 4 further are exact before and after: 315 (0.0014% to 0.0024%), 318 (0.0011% to 0.0013%), 647
   (0.0005% to 0.0007%) and 649 (0.0003% to 0.0010%).
4. **The labels it asks are few except where the candidates are many.** 966 labels in all, median 13.5 a case; 645
   with 413 candidates asks 256 and keeps the pick, 528 asks 140 and 611 116. In 317 the candidates agree on every
   pool input, so it asks none.
5. **Cost.** 0.52 core-hours for the 28 cases, most of it 645's 413 candidates refitted 5 times.

## Interpretation

- #87 found 611 to be a labels problem, and the fix is to let the labels move the candidates: the duel's labels are
  exactly where the networks differ, so refitting on them pulls the true triple's values toward the original while
  the wrong triple cannot follow. Scoring fixed networks on the same labels (#71) tied them.
- It is close to neutral elsewhere: 13 closer and 6 further, all under the line, 24 exact against 22. It was shaped
  after looking at 611, so its rate needs cases it was not shaped on.
- The 200 of 200 depends on running #84's search on 611, which the gate (a missed label) would not do. A search that
  keeps two of the pick's weights (3 pairs and any third parameter, about 240 triples) costs about 3% of #84's and
  includes the true triple whenever the pick has two broken weights, as in 611 (inferred: 611's pick keeps hidden
  weights 10 and 20); running it on every case, with the refit duel, is #89.
