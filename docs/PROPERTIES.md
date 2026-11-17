# Properties a working Titans must have — and the test that guards each

The tests in `tests/` are derived from this ledger, not the other way round.
Each row names where the property comes from, because the original paper is
not the only source: later re-implementations and follow-ups found things the
original did not report. Status is for the PyTorch model in this repo.

Sources: **[T]** Titans, arXiv:2501.00663 · **[R]** Titans Revisited,
arXiv:2510.09551 · **[M]** MIRAS ("It's All Connected"), arXiv:2504.13173 ·
**[A]** ATLAS, arXiv:2505.23735 · **[P]** sili_peridot measurements (JOURNAL.md) ·
**[B]** sili__new/bend_proofs/titans_gate_retention_kernel.

| # | property | source | test | status |
|---|---|---|---|---|
| 1 | One-shot store: from M=0 a write with theta = 0.5/‖k‖² makes `M k = v` exactly | [T] eq 10-13, [P] | [`test_memory.py::test_one_shot_store`](../tests/test_memory.py) | guarded |
| 2 | A write is a rank-1 (outer-product) update | [T] closed-form grad | [`test_write_is_an_outer_product_update`](../tests/test_memory.py) | guarded |
| 3 | Momentum (eta) and forgetting (alpha) each change the update | [T] Table 5 | [`test_momentum_and_forget_change_the_update`](../tests/test_memory.py) | guarded |
| 4 | Write→read is differentiable with a correct gradient | [T] §4 training | [`test_write_then_read_gradcheck`](../tests/test_memory.py) (double, autograd gradcheck) | guarded |
| 5 | State detach cuts the graph (bounded backward cost) | [T] chunking, [P] O(1) | [`test_detach_cuts_the_graph`](../tests/test_memory.py), [`test_state_detached_at_chunk_boundary`](../tests/test_mac.py) | guarded |
| 6 | Knob-free gates follow Law 5: alpha=1/L, eta=1-1/L, theta×(1-eta) | [B], [P] | [`test_kernel_matched_gates_follow_law5`](../tests/test_memory.py) | guarded |
| 7 | Retrieval-lag estimator tracks the true lag | [P] | [`test_lag_estimate_tracks_true_lag`](../tests/test_memory.py) | guarded (needs q/k alignment) |
| 8 | Memory is per batch element; shapes/finiteness over random B, L, d | — | [`test_state_is_per_batch_element`](../tests/test_memory.py), [`test_shapes_and_finiteness`](../tests/test_memory.py) (hypothesis) | guarded |
| 9 | Retrievals are context: chunk tokens see all retrievals, retrievals don't see the chunk | [T] §4.1 MAC | [`test_mac_mask_structure`](../tests/test_mac.py) | guarded |
| 10 | A chunk's writes do not leak into its own logits (retrieve-then-update) | [T] §4.1 | [`test_memory_write_changes_later_logits_only`](../tests/test_mac.py) | guarded |
| 11 | Attention alone solves in-context recall | control | [`test_learns_in_context_recall`](../tests/test_mac.py) (integration) | **measured 1.0000 @400 steps** |
| 12 | Attention alone CANNOT solve beyond-window recall (no leak in the task) | control, [P] | [`test_no_memory_cannot_do_long_gap`](../tests/test_mac.py) (integration) | **measured 0.056 vs baseline 0.071** |
| 13 | With memory, beyond-window recall is learned | [T] S-NIAH/BABILong, [R] "neural memory is the decisive factor" | [`test_memory_learns_long_gap_recall`](../tests/test_mac.py) (integration) | **measured 0.90 (seed 0, 800 steps), 1.0000 (seed 1, 150 steps)** |
| 14 | Writes keyed on the previous token store "key→value", not "token→itself" | [P] retention 0.139→0.803 | [`test_ablation_breaks_long_gap_recall`](../tests/test_mac.py)`[key_on_same_token]` | **pass**: off → 0.383 vs 1.000 with it (base 0.074) |
| 15 | Gradient must reach the write side across the gap (tbptt span ≥ gap) or learning stalls ~0.2-0.3 | [P] frozen arm, this repo | [`test_ablation_breaks_long_gap_recall`](../tests/test_mac.py)`[detach_every_chunk]` | **pass**: detach every chunk → 0.061 (chance) vs 1.000 |
| 16 | Larger chunks help; chunking is what makes Titans lose to baselines | [R] | chunk-size sweep | todo |
| 17 | Persistent (learned, input-independent) tokens: negligible or negative alone | [R] ablation | not implemented; add only with a test showing a gain | todo |
| 18 | Memory updates alone give no meaningful test-time learning after convergence (slow decay past 50 epochs) | [R] | `examples/capacity_sweep.py` (gap axis) | **measured: mild, log-linear in gap and NOT intrinsic** — well-trained seed loses 0.1–0.9 pt per gap doubling (n=2…16), under-trained seed 3–4 pt. Decay = how well BLANK writes were learned to be null (err→0), i.e. a training-quality effect |
| 19 | Matrix memory holds only O(d_k) linearly independent keys | [A] Prop. 1 | `examples/capacity_sweep.py`; [`test_poly2_raises_capacity`](../tests/test_mac.py) | **measured: recovered bindings saturate at C≈16 ≈ d_k/4**; **poly2 raises it: base recovers 17.9–19.2 of 32 bindings, poly2 recovers 32.0/32 (probe saturated; true ceiling > 32, ≥ 1.7×)** (d_k=64; acc·n = 2, 4, 7.6, 12.3, 15, 16). Tail fits acc ≈ C/n (32→0.469, 64→0.250). Under-trained seed saturates at 3.5: C is set by learned key alignment, d_k/4 is the ceiling seen |
| 20 | Single-token online updates memorize tokens, not context; a sliding-window objective (Omega rule) fixes it | [A] | [`test_omega_keeps_capacity`](../tests/test_mac.py), [`test_memory_learns_long_gap_recall[omega]`](../tests/test_mac.py) | window=chunk implemented (1/λ_max-scaled joint step). **Not merely neutral: omega alone recovers 28.3/32 bindings vs 19.2 for the per-token rule (1.5×)** — fitting the window jointly reduces intra-window interference. Context-vs-token memorization on distractor tasks still untested |
| 21 | First-order updates converge to spurious minima; second-order (Muon-style) helps | [A] | — | out of scope for now |
| 22 | L2 attentional bias is one choice among several (ℓp, Huber); retention gate is a regularizer choice (L2, KL, elastic) | [M] | the loss and gate should be swappable; test that swapping preserves #1-#5 | todo |
| 23 | Momentum is what lets deeper (MLP) memories train | [M] | MLP memory + eta sweep | todo (MLP memory not yet in this repo) |
| 24 | Gates must not be left at random init: alpha≈0.5 halves memory each step → below chance | [P] | [`test_default_gates_are_not_random`](../tests/test_mac.py); [`test_ablation_breaks_long_gap_recall`](../tests/test_mac.py)`[alpha_half]` | **pass**: defaults alpha=eta=0, not parameters, seed-independent; alpha 0.5 → 0.061 |
| 25 | Solve RATE across seeds, not a single seed: phase transition timing varies (seed 0: step ~560; seed 1: <150) | [P] reliability, this repo | multi-seed run, report rate | **4/4 seeds > baseline+0.8; 2/4 at 1.0000 by step 150, 2/4 at 0.87-0.90 by 800-1500 (transition timing varies 10x)** |

**Approximate law from the sweep (matrix memory, d_k=64, no forgetting):**
`acc(n, g) ≈ min(1, C_eff/n) − β·n·log2(g/16)` with `C_eff ≤ ~d_k/4 ≈ 16`
set by how orthogonal the learned keys are, and `β ≈ 1e-3` per binding per
gap-doubling for a well-trained model (≈ 1e-2 under-trained). Capacity is
intrinsic; gap decay is not.

Reading the ledger: every "guarded" row is a fast unit test. "measured"
rows are integration tests (`pytest -m integration`). "todo" rows are
properties the literature says matter that nothing here checks yet — the
first ones to add are #14/#15 (ablations of the two things that made #13
work) and #19 (capacity cliff), since the kernel port must preserve them.

## Family matrix (A / B / C / B+C), ledger rows 26-29

| # | property | test | status |
|---|---|---|---|
| 26 | A (attention only) cannot do beyond-window recall | [`test_no_memory_cannot_do_long_gap`](../tests/test_mac.py) | measured 0.056 vs baseline 0.071 |
| 27 | B (RMT) learns beyond-window recall through passed tokens alone | [`test_family_learns_long_gap_recall[rmt]`](../tests/test_mac.py) | **FAILS, strict-xfail. Measured 0.33±0.01 for m=4 and m=16, with and without a 0→4→8→16 gap curriculum (4 runs). Per-pair accuracy uniform across write order (0.31–0.36): it carries the SET of written values (1/3 ≈ guessing among them) but never binds key→value.** |
| 28 | B+C (Combined) learns it | [`test_family_learns_long_gap_recall[combined]`](../tests/test_mac.py) | **passes** (800 steps) |
| 29 | Bindings recovered per family at 32 live, d=64 | [`test_family_capacity_probe`](../tests/test_mac.py) | **rmt 1.5 (train acc 0.15 — cannot learn 8 pairs), titans 19.4, combined 23.7.** B+C > C by ~4: the state tokens add capacity when an associative store exists, even though alone they cannot bind. The prediction "B ∝ m·d" is FALSIFIED at this scale — B's limit is binding, not slots. |

**Reading of the family matrix.** C stores functions (an outer product k vᵀ *is*
a binding); B stores a point and must *learn* to bind inside it through BPTT,
and at 2 layers / 3000 steps it does not — it learns the bag of values and
guesses. That is the A/B/C taxonomy's prediction showing up empirically, and
it is the reason the sili substrate (memory as weights) is the bet. Open: does
B bind with more depth, a cross-attention write head, or m·d ≫ bindings? Not
pursued now; C and B+C are the path.

| 30 | B+C beats C under long training | `docs/results/2026-11-16-equal-steps-64-pair-probe.md`, `docs/results/2026-11-16-seeds-and-poly2-at-scale.md` | **combined leads**: over seeds that trained (0, 2, 3) 56.3 vs 54.7 of 64 (30–40k windowed), worst eval 53.7 vs 43.4; 30.7 vs 30.2 of 32. Seed 1 never trains for EITHER family (see row 32). |

**Long-run method note.** Equal wall-clock gave unequal steps (titans 45k on
the 1660 Ti, combined 81k on the 2060, rmt 450k on the P40): compare by step,
or budget by steps. Peak temperatures under the guard: 1660 Ti **75 °C**
(closest to its abort), P40 51 °C at a 150 W cap, 2060 45 °C.
| 31 | poly2 + omega trains at scale (128-key, 16 pairs, d=64) | same results files | **FAILS** together (train acc 0.74 at 40k). **poly2 alone PASSES**: 1.00 train by 1.5k, 58.4 of 64 at 4k (1 seed). omega alone at this spec not yet measured. |
| 32 | Training escapes the start for every seed | `docs/results/2026-11-16-seeds-and-poly2-at-scale.md` | **FAILS**: 2 of 8 runs (seed 1, both families) never train in 40k (combined collapses to 1 distinct prediction); titans s3 plateaus to ~10k. No test yet; `--seed` sets data and init together. |
