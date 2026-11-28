# GameScaling-0.8B: where training memory goes (2026-11-23)

Run on the P40 with fp32, the text path, batch 1, one forward + backward with
a next-token loss. transformers falls back to its pure-PyTorch *chunked*
gated delta rule (`torch_chunk_gated_delta_rule`, 64-token chunks), because
flash-linear-attention is not installed. The fallback applies in training,
not only in inference. Script: `examples/train_memory_breakdown.py`.

The weights take 2870 MiB (tied embeddings).

| T (tokens) | memory after forward, above weights | peak in backward, above weights |
|--:|--:|--:|
| 256 | 2260 MiB | 5072 MiB |
| 512 | 4499 MiB | 5472 MiB |
| 1024 | 8967 MiB | 10907 MiB |

**What grows per token,** from the 512 → 1024 difference:
- 8.7 MiB per token for the whole model, about 0.36 MiB per token per layer.
- Bytes saved for backward, per extra token (this double-counts views that
  share storage):
  - gated delta net (18 linear-attention layers): about 6.6 MiB, about 377
    KiB per layer. This is the largest share.
  - MLP (24 layers): about 1.5 MiB, about 64 KiB per layer.
  - full attention (6 layers): about 0.56 MiB, about 93 KiB per layer.
  - other: about 2.5 MiB, mostly the fp32 logits that cross-entropy keeps
    (248k vocab × 4 B, about 0.95 MiB per token), plus norms.
- The lm_head's 971 MiB "saved" is the tied embedding weight, which is
  already resident. It is not activation memory.

**Corrections to earlier numbers:**
- The "1 MiB per token per layer" used before was about 3× too high for this
  setting.
- The fallback stores the recurrent state once per 64-token chunk, not at
  every token. The per-token cost comes from the chunk algorithm's
  intermediates.

**What this means for the P40 (24 GB):**
- Fixed fp32 cost: weights + gradients + AdamW ≈ 2.9 + 2.9 + 5.7 ≈ 11.5 GiB.
- Activations at 8.7 MiB per token:
  - 1k tokens: about 8.7 GiB, about 20 GiB in total, which fits;
  - 2k tokens: about 17.5 GiB, about 29 GiB in total, which does not fit
    without e-BPTT chunks, checkpointing or pipelining.
- Backward adds about 2 GiB over the forward peak at 1k tokens.

**Not measured:** flash-linear-attention. Its chunk kernel has its own
backward and recomputes the within-chunk intermediates instead of saving
them, so it should cut the gated-delta-net share, the largest one. By how
much is not measured. It needs Triton; whether Triton runs on the P40 (sm_61)
is unconfirmed, and the 2060 (sm_75) should work.
