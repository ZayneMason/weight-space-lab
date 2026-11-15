# The weight edit that taught a model no math

*Draft, November 15, 2026. Not yet published.*

A published weight edit lifts Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1%. That is a
40-point jump from changing about 130,000 numbers in two layers, with no training. It is
easy to read that as "the edit made the model better at math".

The transcripts say something else. On the problems where the original model finished its
reasoning, the edited model is no better: 11 correct against 12. The entire gain comes
from problems where the original got stuck in a repetition loop and never finished. The
edit did not teach the model any math. It unstuck it, and the model then solved the
problems on its own.

## The setup

The edit is called **p50c**. It comes from Takehi et al.'s weight-space programming
experiments, where a language model was asked to write edits to another model's weights
directly. p50c removes what the authors describe as a "loop direction" from 64 MLP output
columns in each of layers 15 and 19 of Qwen3.5-0.8B. My lab reproduces it exactly: 130,188
BF16 entries change, both edited tensors match the published hashes, and the other 318
tensors are bitwise unchanged.

The benchmark is MATH-500, run as upstream ran it: 500 problems, 4 samples each, a
32,768-token ceiling, the same sampling for every arm. Upstream also tested two weaker
edits, **p10c** (loop-direction removal from 30 matrices) and **p25c** (one attention head
switched off).

I did not run these evaluations. Upstream released its full records: per-sample
correctness for all 2,000 samples per arm, plus the full text of sample 0 for problems
0 to 199. My code reproduces every published aggregate from those records, and my grader
agrees with the recorded grade on 800 of 800 transcripts.

| Arm | Accuracy | Hit the 32,768-token ceiling | Gain over unedited |
|---|---:|---:|---:|
| unedited | 7.90% | 93.0% | — |
| p10c | 12.35% | 87.3% | +4.45 |
| p25c | 14.10% | 86.4% | +6.20 |
| p50c | 48.10% | 33.1% | **+40.20** [36.85, 43.55] |

The ceiling column is the first clue. The unedited model runs out of tokens on 93% of its
samples. p50c runs out on a third.

## The method: pair the transcripts, split by who finished

Each transcript gets a few measurements taken from its text alone:

- **Closed:** the model wrote `</think>`, so it ended its reasoning.
- **Found:** some `\boxed{}` answer anywhere in the text, reasoning included, matches the
  gold answer.
- **Loop:** the last 20,000 characters compress more than 10× with zlib. A model repeating
  itself compresses very well.

Then I pair the unedited and p50c transcripts problem by problem and split the 200
problems into four cells by which model closed its reasoning. The accuracy difference is
exactly the sum of what each cell contributes, so the split shows where the 40 points come
from. Intervals are 95% bootstraps over problems.

## Where the gain lives

On the 200 paired problems, p50c gains 38.5 points [31.5, 45.5]. Here is the split:

| Unedited closed? | p50c closed? | Problems | Unedited correct | p50c correct | Contribution |
|---|---|---:|---:|---:|---:|
| yes | yes | 12 | 12 | 11 | −0.5 |
| yes | no | 1 | 1 | 0 | −0.5 |
| **no** | **yes** | **140** | **2** | **81** | **+39.5** |
| no | no | 47 | 0 | 0 | 0.0 |

One cell carries the result. On the 140 problems where the original never finished and
p50c did, the original got 2 right and p50c got 81. That cell accounts for 103% of the
gain; the other cells net slightly negative.

Where both models finished, the edit adds nothing. That is 12 problems, and the original
solved all 12.

## The original wasn't done thinking. It was stuck.

A natural first guess is that the original had the answer and just couldn't stop talking.
The data rules that out.

- **185 of its 187 unfinished samples end in a repetition loop.** That share stays between
  98.4% and 98.9% for loop thresholds from 0.05 to 0.2, so it does not hinge on the cutoff.
- **The loop starts early,** at a median of 22% of the way through the text. The rest of
  the token budget is spent repeating.
- **In 78 of the 81 rescued problems, the original never wrote the right answer** in any
  `\boxed{}` across its 32,768 tokens.

So the original was not withholding answers it had already reached. It fell into a loop
before getting there. With p50c it escapes, keeps reasoning, and lands on the answer. On
the rescued problems p50c solved, its transcripts have a median length of about 39,000
characters, roughly half the original's 75,000 for the same problems.

The two weaker edits show the same pattern at smaller scale. p10c rescues 31 problems
(1 to 23 correct) and p25c rescues 20 (1 to 16). On problems both models finished, each
edit scores exactly the same as the original. Each also stops finishing 3 or 4 problems the
original had solved and loses 3 correct answers.

p50c does not eliminate loops; some of them move. 42 of its 200 samples close their
reasoning but never give a boxed answer, and 26 of those loop *after* `</think>`, for
example repeating "Wait, I will provide the answer…". A lenient grader would recover only 4
of them, so missing boxes cost about 2 points rather than hiding a bigger gain.

## Why this matters for reading weight-edit results

A benchmark number blends at least two things: whether the model can do the task, and
whether it gets to the end of its output in a usable form. For small reasoning models with
long chains of thought, the second can dominate. Here it accounts for essentially the whole
40 points.

That changes what the edit is. p50c is best described as an **anti-loop edit**, which fits
upstream's own framing of it as removing a loop direction. Calling it a math improvement
would claim a capability gain the data does not show.

The practical lesson for anyone evaluating weight edits, steering vectors or other
surgical changes: before crediting a gain to capability, split the result by whether the
model finished. If the gain lives entirely in the "used to not finish" cell, you have found
a fix for a failure mode, and the right comparison is against other ways of fixing that
failure, like repetition penalties or a different decoding setup. *(That comparison is my
suggestion; I have not run it.)*

## Caveats

- **One model, one benchmark, upstream's data.** These are Takehi et al.'s released records,
  not a run from my lab.
- **200 transcripts, not 2,000.** Text exists only for sample 0 of problems 0 to 199, a fixed
  subset. Its accuracy is close to the full set's (unedited 7.5% vs 7.9%, p50c 46.0% vs
  48.1%).
- **The "no change when both finish" claim rests on 12 easy problems.** The original solves
  all of them. This cannot rule out a reasoning gain on harder problems.
- **What the original would have done past its loop is unobservable,** so some reasoning
  change on the rescued problems cannot be fully excluded.
- **"Never wrote the right answer" counts only boxed answers.** An original trace that
  reached the right value without boxing it counts as never writing it, so 78 of 81 is an
  upper bound.
- **"Closed" is read from the text,** not the engine's finish reason.
- **Token-budget curves are not run yet.** The code is written and tested, but it needs the
  Qwen tokenizer, which I could not download in the environment I used.

## What's next

1. **Token-budget curves.** Accuracy as a function of tokens spent, for both models. If
   p50c's advantage shrinks to nothing at matched budgets on problems the original finishes,
   that confirms the story from a second angle.
2. **The lab's own paired GPU run,** with stored token IDs, so the analysis does not depend
   on upstream's 200-transcript subset.
3. **Loop-breaking baselines.** Compare p50c against a repetition penalty or loop-aware
   decoding on the unedited model. If a decoding change recovers most of the 40 points, the
   edit is a weight-space way of doing something decoding already does.

Code and evidence: `experiments/p50c-termination/`, `qwen/termination.py` and
`evidence/qwen-termination/` in
[weight-space-lab](https://github.com/ZayneMason/weight-space-lab). Upstream records:
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), revision
`4d09ba6`, CC BY 4.0.
