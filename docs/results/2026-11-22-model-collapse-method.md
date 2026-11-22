# Model collapse / forgetting with neuron energy: method and feasibility (2026-11-22)

Written before the pilot's results. Script: `examples/model_collapse.py`.

## What the literature does

| paper | models | loop | sampling | metric | finding |
|---|---|---|---|---|---|
| Shumailov et al., Nature 2024 ([arXiv](https://arxiv.org/abs/2305.17493)) | OPT-125M, fine-tuned on WikiText-2 | 64-token real prompt, model writes 64, next generation trains on that, same size; 5 runs | 5-way beam | perplexity on real test text; model 0's perplexity on the samples | perplexity rises, samples repeat; repetition penalty 2.0 made it worse; 10 % real data only softens it |
| Gerstgrasser et al. 2024 ([arXiv](https://arxiv.org/abs/2404.01413)) | 9M to 125M, new init each time | replace vs accumulate | T 1.0, 0.3 | test cross-entropy | replace collapses (faster at T 0.3), accumulate does not |
| Drayson et al., EMNLP 2025 ([arXiv](https://arxiv.org/abs/2502.15654)) | GPT-2 124M, SmolLM2 135M to 1.7B | WikiText-2, 256+256, loss on continuation, AdamW 1e-5, batch 8, 1 epoch, 10 generations | greedy, beam, T .9, 1.0, top-k, nucleus | perplexity, distinct n-grams, MAUVE | greedy/beam worst; **T = 1 on SmolLM2-360M barely degrades (13.96 to 15.39)**; T .9 is the most repetitive stochastic one |
| Maiti et al. 2025 ([arXiv](https://arxiv.org/abs/2512.06256)) | 12B and 13B, talking, no training | 25 turns, 50 tokens | top-p .95, T .7 | embedding / Jaccard / BLEU distance | 35 of 50 conversations end in a repetition loop |

- Shumailov et al. separate collapse (a chain of models misperceiving the data)
  from catastrophic forgetting (one model losing what it knew). A student that
  is never reset shows both, so that is what the pilot uses.
- None of these found trains two models on their conversation, or uses neuron
  energy. That part is ours.
- "T = 1 collapses" (TODO 2c's expected control) is not safe to assume at this
  model size, so greedy and T .9 are the collapse controls.

## Pilot

- 2060 = teacher (fp16), P40 = student (fp32); GameScaling-0.8B work copy.
- Generation 0: one epoch on real text. Generations 1 to 10: the teacher writes
  128 × 64-token continuations, the student trains one epoch on them, and the
  teacher takes the student's weights.
- Energy on both models: MLP activations of layers 2, 6, 10, 14, 18, 22; p = 1;
  drive = cost × mean |h|; decay = drive/4; reset to zero on fire; starts at
  rest; accumulates across batches and generations. Student loss = LM + 1e-3 ×
  aux per token. Evaluation always has energy off.
- Arms: real control (retrain on the same real text), greedy, T .9, T 1.0, and
  energy cost .01 / .1 on greedy and on T .9. One seed: a pilot, not a verdict.

**Forgetting prevented**, fixed in advance: with
`rise = ln(test perplexity at generation g) − ln(at generation 0)`,

    prevented = 1 − rise(energy) / rise(same sampling, energy off)

1 = no forgetting, 0 = no help, negative = energy made it worse. The real
control's rise is the part that is just over-training on 128 samples. Report it
next to distinct-3-gram, looping and rare-token mass, because diversity alone
can rise while quality falls (Drayson et al.).

## Feasibility measured first

- 752M text parameters load with no missing keys. fp16 (2060) and fp32 (P40)
  perplexity: 35.435 and 35.433. Cached and full passes agree on argmax.
- P40, fp32, AdamW: 1.25 s/step at 8 × 128 tokens, 20.0 GiB. 8 × 512 runs out
  of memory: backprop keeps the linear-attention state of every token (about
  1 MiB per token per layer), which is where e-BPTT would apply.
- 2060, fp16: 339 tokens/s at batch 8, 3.8 GiB peak.
- Energy on one layer is nearly inert below cost .1. On the six layers it has a
  graded effect that accumulates with tokens seen (perplexity against energy
  off, at 256 / 768 tokens): .01 gives 1.10 / 1.22, .1 gives 1.25 / 1.32,
  1.0 gives 1.53 / 1.56, with 0.1 to 3 % of activations changed.
