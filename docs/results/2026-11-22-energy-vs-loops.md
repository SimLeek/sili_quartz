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
- **Combining sites is worse than either alone.** There is no dose at which
  all three break loops without gibberish.
- The CPU training path works with all three sites (gradients reach all 320
  parameters, all finite). Peak memory rose 12.3 → 16.6 GiB on CPU for 4 × 128
  tokens; GPU still to measure.

Next: the site sweep (`--suite sites`), queued behind the pilot, with
damage-matched noise arms and the student's free-running repetition logged
every generation.
