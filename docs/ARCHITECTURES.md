# Recurrent transformers: which are different architectures, which are knobs

Sorted by *what the memory physically is*, because that is what fixes the
properties. Paper names are secondary.

| family | memory is… | written by… | trained by… | examples |
|---|---|---|---|---|
| **A. Cached activations** | past hidden states / KV, kept verbatim | nothing — it is a cache | no gradient into the cache (stop-grad) | Transformer-XL |
| **B. Passed state tokens** | a few vectors (m "memory tokens") carried segment to segment | the backbone itself, via special positions | BPTT across segments (truncated) | RMT; HMT = RMT + a retrieval bank |
| **C. Memory as weights** | a matrix or MLP, updated at test time by an inner optimizer on an associative loss | gradient steps on ‖M(k) − v‖ (or another loss) | outer loop learns the projections and gate nets; inner loop is the memory | Titans, Moneta, Yaad, Memora, ATLAS |

## A — Transformer-XL (2019)

Segment-level recurrence: the previous segment's hidden states are cached
(detached) and offered as extra keys/values, with relative positions.
Nothing is learned *about* the memory; capacity grows linearly with the
cache; information is verbatim and decays only by eviction. Useful as the
"recent tokens" tier. Cannot generalize over what it saw, only re-read it.
**Different architecture class** from B and C.

## B — RMT (2022), HMT (2024)

RMT: m learned memory tokens are prepended (read) and appended (write) to
each segment; the write positions' outputs become the next segment's
memory. Fixed size, compressive, everything the backbone chooses to
write. Gradient crosses segments by truncated BPTT, so what it can learn
to carry is bounded by the unroll length — exactly the lesson re-measured
in this repo (`tbptt_chunks` must span the gap or learning stalls).

HMT is RMT plus two tiers: a **sensory** tier (last k=32 token embeddings
of the previous segment prepended, i.e. a tiny Transformer-XL cache) and a
**recall** tier (a sliding window of N=300 past memory embeddings,
retrieved by softmax cross-attention without a value projection, queried
by a learned-prompt summary of the first half of the segment). Plug-and-
play on a pretrained backbone, +0.5–1.3% params, unrolled up to 15
segments. So HMT is a *composition* (A + B + retrieval over B's history),
not a new mechanism. ARMT (2025) adds a per-layer delta-rule store, which
is C bolted onto B.

## C — Titans and its knobs: Moneta, Yaad, Memora, ATLAS

All five share one skeleton: project k, v, q from the layer input; keep a
memory M; per step, take an optimizer step on an associative loss; read
with M(q); combine with attention (MAC/MAG/MAL). MIRAS makes the knobs
explicit — four axes, every one a few lines of code:

| axis | Titans | Moneta | Yaad | Memora | ATLAS |
|---|---|---|---|---|---|
| memory | matrix or MLP | 2-layer MLP | 2-layer MLP | 2-layer MLP | residual 2-layer MLP |
| loss ("attentional bias") | L2 | ℓ3 | Huber | L2 | L2 |
| retention gate | L2 decay (alpha) | ℓq + L2 (q=4) | L2 local+global | KL on a softmax simplex | L2 decay |
| inner optimizer | GD + momentum (eta) | GD | GD | GD | **Muon** (Newton-Schulz on the momentum) |
| update context | 1 token (online) | 1 token | 1 token | 1 token | **Omega rule: window of c tokens, weighted** |
| key feature map | identity | identity | identity | identity | **polynomial φ_p(k), capacity O(d_k) → O(d_k^p)** |

Moneta / Yaad / Memora are therefore **the same architecture with a
different inner loss and regularizer** — not worth separate model classes.
ATLAS adds three real mechanisms, all inside the memory update, none in
the transformer: the windowed objective (memorize *context*, not tokens —
the one that matters for an agent), the polynomial feature map (capacity),
and Muon (better inner optima). Its "DeepTransformers/Dot" backbone is a
separate claim about replacing attention with the same machinery; we keep
attention.

## What this means for sili_quartz

Three layers, as you proposed, and they map onto the families:

1. **base transformer** — chunked causal attention. Exists (`memory=False`).
2. **recurrent transformer** — the A/B tier: a Transformer-XL cache of the
   previous chunk's K/V (cheap, verbatim, short range) and optionally RMT
   memory tokens. Titans Revisited found persistent tokens alone negligible,
   so RMT tokens are the last thing to add, with a test that shows a gain.
3. **memory as weights with every ATLAS switch** — `TitansMemory` grows
   four options, each independently testable against the ledger:
   `loss={l2,lp,huber}`, `retention={l2,lq,kl}`, `optimizer={gd,momentum,muon}`,
   `window=c` (Omega; c=1 is Titans), `features={identity,poly_p}`.

Two of those earn their place first, for the game/programming-agent goal:

- **Omega window (c = chunk).** One gradient over all (k_i, v_i) of the
  chunk instead of a per-token loop. Memorizes the chunk as a pattern, not
  as 8 unrelated tokens; and it removes the sequential loop, which is also
  what makes the CUDA port a batched matmul instead of a scan.
- **Polynomial features on keys.** Capacity is the cliff ATLAS proved
  (O(d_k) pairs for a matrix memory); a game state has far more than d_k
  things worth binding.

Muon and the alternative losses go in after, each behind a test from
`PROPERTIES.md`, because the ledger's rows 1–5 must keep holding for
every switch setting.
