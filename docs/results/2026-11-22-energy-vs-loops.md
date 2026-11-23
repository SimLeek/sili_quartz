# Why energy did not stop repetition, and why perplexity can't see it (2026-11-22)

Follow-up to a smoke run of the collapse pilot (2 generations, 16 samples).
CPU diagnostics on GameScaling-0.8B: one repeated phrase, two real-text
segments, one run each. Diagnostic, not a verdict.

## Scoring generated text by perplexity rewards repetition

Two different perplexities, and this section is about the second:
- **test perplexity** (every results table): the student's perplexity on fixed
  real test text, teacher-forced. Training on loops makes it *worse*.
- **original_on_samples** (a structure check): the unmodified model's
  perplexity on the *generated* samples. Loops are easy to predict, so they
  score *low*.

Original model's perplexity on the generated samples, all-layer smoke run:

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

## Energy on attention and the residual (user, 2026-11-22)

`sili_quartz/energy_hooks.py` puts energy on any module output. Each site is
scaled to its layer's MLP activations (means: MLP 0.21, attention 0.027,
residual 0.098), so one cost is the same dose everywhere. Without that scaling
a residual that fires is set to about 40x its usual size: `resid .1 unscaled`
below turns into gibberish with only 0.3 % of values changed.

Same method as above (decay drive/40). "Loop" = the free-running continuation
of a repeated phrase; distinct-3 under 0.5 means it is still repeating (the
pure loop scores 0.17). Damage = extra nats on the last 64 tokens of real text
(mean of two segments).

| arm | breaks loop 1? (distinct-3) | damage to real text | its own real continuations |
|---|---|---|---|
| off | no (0.17) | 0 | fine |
| mlp .01 / .03 | no (0.17, 0.26) | 0.69 / 1.01 | fine |
| **mlp .05** | **yes (0.70)**, into a list of years | **1.37** | fine |
| mlp .1 | yes (0.80), a new short loop | 1.59 | degraded |
| attn .01 | no (0.17) | 0.66 | some repetition |
| **attn .02** | **yes (0.87)**, garbled words | **1.46** | spaces lost |
| attn .03 / .1 | yes, gibberish | 2.95 / 6.87 | gibberish |
| resid .003 / .01 | no (0.17, 0.26) | 0.50 / 2.33 | spaces lost at .01 |
| resid .1 | gibberish | 9.26 | gibberish |
| all .001 / .003 | no | 0.15 / 0.72 | fine |
| all .01, mlp+attn .02 | yes, gibberish | 3.3 to 3.6 | gibberish |
| noise .3 / .4 | no (0.17, 0.24) | 0.80 / 2.07 | fine / numbers |
| noise .5 | yes (0.96), gibberish | 4.12 | gibberish |

Repeated on two other loops and a third real segment (mlp .05, attn .02,
noise .35, noise .4):

| arm | loop 2 (short phrase) | loop 3 (20-token sentence) | real C damage, continuation |
|---|---|---|---|
| mlp .05 | breaks, coherent text | still repeats (0.50) | 1.04, **falls into a "= =" loop** |
| attn .02 | breaks, garbled | breaks (0.98), half coherent | 1.26, fine |
| noise .35 | repeats (0.24) | repeats (0.57) | 1.36, fine |
| noise .4 | repeats (0.41) | partly (0.80) | 2.25, numbers |

What it shows (one run per prompt, so read it as direction, not size):

- **Energy is more specific than noise.** At ~1.0 to 1.5 nats of damage,
  energy on MLP or attention breaks short loops. Noise at the same damage
  doesn't, and needs 2 to 4 nats.
- **Attention fits the guess.** Per unit of damage to real text it pushes
  hardest on the loop (attn .02: 1.45 nats on the loop vs 1.46 on real text;
  MLP .05: 0.44 vs 1.37; noise .4: 0.46 vs 2.07). But it garbles words.
- **MLP alone can break loops** once decay lets energy build up, which
  contradicts my earlier guess. It can also *create* a loop on real text.
- **The residual is the worst site:** no loop break before it drops spaces.
- **All three sites together** found no dose that breaks loops without
  gibberish. MLP + attention at lower doses does (below).
- The CPU training path works with all three sites (gradients reach all 320
  parameters, all finite). Peak memory rose 12.3 → 16.6 GiB on CPU for 4 × 128
  tokens; GPU still to measure.

Next: the site sweep (`--suite sites`), queued behind the pilot, with
damage-matched noise arms and the student's free-running repetition logged
every generation.

## MLP + attention, and token temperature (same day, follow-up)

Every arm is scored on its own free-running 48 tokens. Loops: distinct-3 on
three repeated phrases (under 0.5 = still repeating). Real text: distinct-3 and
the unmodified model's perplexity on what was written (the true next 48 tokens
score 8 to 20; loops score near 2; gibberish in the thousands). Temperature
arms are the mean of 3 seeds; everything else is one run.

| arm | loops broken (distinct-3 of 3 loops) | real text: distinct-3 / perplexity of what it wrote |
|---|---|---|
| greedy | 0 of 3 (0.17, 0.09, 0.46) | 0.94-1.0 / 2.4-3.9 |
| T 0.7 | 0 of 3 | 0.95-1.0 / 3.9-8.6 |
| T 1.0 | 0 of 3 (loop 1: 0.45) | 0.97-1.0 / 17-47 |
| T 1.3 | 2 of 3, as gibberish | 1.0 / 11,600-27,000 |
| repetition penalty 1.2 | 0 of 3 | 0.94-1.0 / 2.9-6.2 |
| repetition penalty 1.5 | 1 of 3 (the long sentence) | 0.98-1.0 / 8-23 |
| mlp .05 | 2 of 3 | 0.17-1.0 / 1.7-11 (one real passage looped) |
| attn .02 | 3 of 3, garbled words | 0.80-1.0 / 5.8-30 |
| **mlp .03 + attn .01** | **3 of 3 (1.0, 0.91, 1.0), into readable text** | **0.87-0.91 / 42-89** |
| mlp .02 + attn .01 | 2 of 3 | 0.57-1.0 / 7.7-55 |
| mlp .03 + attn .005 | 1 of 3 (0.70, 0.37, 0.78) | 0.22-1.0 / 3.6-13 |
| mlp .05 + attn .01 | 2 of 3, with markup | 0.41-0.94 / 10-97 |
| mlp .01 + attn .015 | 3 of 3, one garbled | 0.15-0.87 / 3.3-43 (one real passage looped) |

- Token temperature cannot break these loops: the model puts ~0.998 on the
  next loop token, so sampling at T ≤ 1 keeps it. T 1.3 breaks them only by
  destroying everything else, as in TODO 2c.
- A repetition penalty is the strongest token-level baseline here, and it
  broke only the loop with the longest period.
- Energy carries state, so it can push against a loop that sampling can't.
  mlp .03 + attn .01 broke all three into readable text, at a real-text cost
  a bit above T 1.0 (perplexity 42-89 vs 17-47). That makes it the best pair
  found, not a tuned optimum: neighbouring doses are erratic, one run each.
- The site sweep now includes this pair, repetition penalty 1.5 and T 1.0.
