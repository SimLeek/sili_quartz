# Why energy did not stop repetition, and why perplexity can't see it (2026-11-22)

Follow-up to a smoke run of the collapse pilot (2 generations, 16 samples).
CPU diagnostics on GameScaling-0.8B: one repeated phrase, two real-text
segments, one run each. Diagnostic, not a verdict.

## Perplexity rewards repetition

Original model's perplexity on the samples, all-layer smoke run:

| samples | generation | perplexity | share looping |
|---|---|---|---|
| real text | 1 / 2 | 50.6 / 27.3 | 0 / 0 |
| greedy | 1 / 2 | 2.9 / 2.2 | 19 % / 63 % |
| greedy + energy .1 | 1 / 2 | 5.6 / 2.2 | 6 % / 94 % |

Loops score 10x better than real text. Test perplexity on real text is blind
the other way: it is teacher-forced, so a model that loops whenever it runs
free can still score well (energy .1, generation 2: 31.1 against greedy's 48.2,
with 94 % of its samples looping). Judge collapse on free-running repetition
and diversity next to perplexity.

Energy did *reduce* looping at generation 1 (6 % vs 19 %) before it was worse at
generation 2. n = 16, so either could be noise.

## The anti-loop push, and what cancels it

With decay, energy settles at `cost·(m − |h|)/decay` (m = the mean it is
calibrated to). It reaches ±2 only if |h| stays outside `m·(1 ± 2·decay/drive)`.
The pilot's decay = drive/4 makes that band ±50 %, and 93.5 % of neurons average
inside it on real text: they can never fire or shut off from a steady input.

Method: 128 tokens of real text, then 384 of either a repeated phrase or more
real text. Cost .1, all 24 layers' MLP neurons.

| energy | extra nats on the loop, early / mid / late | extra nats on real text, late (A, B) | greedy still follows loop | neurons off: loop / real A / real B | same neurons off in two unrelated texts |
|---|---|---|---|---|---|
| layer mean, decay drive/4 (the pilot) | .48 / .24 / .52 | 1.30, 1.19 | 83 % | 2.9 / 1.8 / 1.9 % | 52 % |
| layer mean, drive/40 | .46 / .90 / 1.44 | 1.71, 1.78 | 54 % | 8.4 / 7.0 / 7.3 % | 53 % |
| layer mean, no decay | .53 / .99 / 1.32 | 2.09, 1.63 | 67 % | 9.4 / 8.3 / 8.6 % | 54 % |
| per-neuron mean, drive/4 | .04 / .02 / .03 | .38, .07 | 100 % | 1.2 / .17 / .09 % | 4 % |
| per-neuron mean, drive/40 | .05 / .03 / .07 | .73, .47 | 100 % | 4.9 / 2.3 / .6 % | 5 % |
| per-neuron mean, no decay | .04 / .03 / .04 | .81, .49 | 100 % | 6.0 / 3.2 / .8 % | 7 % |

What was measured:

- **The pilot's decay removes the build-up.** With drive/40 or no decay the
  push on the loop grows about 3x as it repeats. With drive/4 it does not.
- **A per-layer mean mostly switches off the same neurons whatever the text:**
  about half are shared between unrelated texts, so much of it is a fixed
  lesion. It costs real text more than it costs the loop.
- **A per-neuron mean detects repetition** (2 to 13x more neurons off on the
  loop than on real text, few shared between texts) **but does not break it:**
  greedy follows the loop 100 % of the time.

Not tested (hypothesis): copying runs through attention and the residual
stream, so MLP neurons alone can notice a loop without being able to stop it.
