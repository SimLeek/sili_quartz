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
| 1 | One-shot store: from M=0 a write with theta = 0.5/‖k‖² makes `M k = v` exactly | [T] eq 10-13, [P] | `test_memory.py::test_one_shot_store` | guarded |
| 2 | A write is a rank-1 (outer-product) update | [T] closed-form grad | `test_write_is_an_outer_product_update` | guarded |
| 3 | Momentum (eta) and forgetting (alpha) each change the update | [T] Table 5 | `test_momentum_and_forget_change_the_update` | guarded |
| 4 | Write→read is differentiable with a correct gradient | [T] §4 training | `test_write_then_read_gradcheck` (double, autograd gradcheck) | guarded |
| 5 | State detach cuts the graph (bounded backward cost) | [T] chunking, [P] O(1) | `test_detach_cuts_the_graph`, `test_state_detached_at_chunk_boundary` | guarded |
| 6 | Knob-free gates follow Law 5: alpha=1/L, eta=1-1/L, theta×(1-eta) | [B], [P] | `test_kernel_matched_gates_follow_law5` | guarded |
| 7 | Retrieval-lag estimator tracks the true lag | [P] | `test_lag_estimate_tracks_true_lag` | guarded (needs q/k alignment) |
| 8 | Memory is per batch element; shapes/finiteness over random B, L, d | — | `test_state_is_per_batch_element`, `test_shapes_and_finiteness` (hypothesis) | guarded |
| 9 | Retrievals are context: chunk tokens see all retrievals, retrievals don't see the chunk | [T] §4.1 MAC | `test_mac_mask_structure` | guarded |
| 10 | A chunk's writes do not leak into its own logits (retrieve-then-update) | [T] §4.1 | `test_memory_write_changes_later_logits_only` | guarded |
| 11 | Attention alone solves in-context recall | control | `test_learns_in_context_recall` (integration) | **measured 1.0000 @400 steps** |
| 12 | Attention alone CANNOT solve beyond-window recall (no leak in the task) | control, [P] | `test_no_memory_cannot_do_long_gap` (integration) | **measured 0.056 vs baseline 0.071** |
| 13 | With memory, beyond-window recall is learned | [T] S-NIAH/BABILong, [R] "neural memory is the decisive factor" | `test_memory_learns_long_gap_recall` (integration) | **measured 0.90 (seed 0, 800 steps), 1.0000 (seed 1, 150 steps)** |
| 14 | Writes keyed on the previous token store "key→value", not "token→itself" | [P] retention 0.139→0.803 | covered by #13's config (`write_key_from_prev=True`); needs its own ablation test | todo |
| 15 | Gradient must reach the write side across the gap (tbptt span ≥ gap) or learning stalls ~0.2-0.3 | [P] frozen arm, this repo | covered by #13's config (`tbptt_chunks` ≥ episode); needs its own ablation test | todo |
| 16 | Larger chunks help; chunking is what makes Titans lose to baselines | [R] | chunk-size sweep | todo |
| 17 | Persistent (learned, input-independent) tokens: negligible or negative alone | [R] ablation | not implemented; add only with a test showing a gain | todo |
| 18 | Memory updates alone give no meaningful test-time learning after convergence (slow decay past 50 epochs) | [R] | long-stream drift test | todo |
| 19 | Matrix memory holds only O(d_k) linearly independent keys | [A] Prop. 1 | capacity sweep: n_pairs vs d, expect a cliff | todo |
| 20 | Single-token online updates memorize tokens, not context; a sliding-window objective (Omega rule) fixes it | [A] | compare per-token vs window write on recall with distractors | todo (research) |
| 21 | First-order updates converge to spurious minima; second-order (Muon-style) helps | [A] | — | out of scope for now |
| 22 | L2 attentional bias is one choice among several (ℓp, Huber); retention gate is a regularizer choice (L2, KL, elastic) | [M] | the loss and gate should be swappable; test that swapping preserves #1-#5 | todo |
| 23 | Momentum is what lets deeper (MLP) memories train | [M] | MLP memory + eta sweep | todo (MLP memory not yet in this repo) |
| 24 | Gates must not be left at random init: alpha≈0.5 halves memory each step → below chance | [P] | init test: default gates are not random | todo |
| 25 | Solve RATE across seeds, not a single seed: phase transition timing varies (seed 0: step ~560; seed 1: <150) | [P] reliability, this repo | multi-seed run, report rate | **4/4 seeds > baseline+0.8; 2/4 at 1.0000 by step 150, 2/4 at 0.87-0.90 by 800-1500 (transition timing varies 10x)** |

Reading the ledger: every "guarded" row is a fast unit test. "measured"
rows are integration tests (`pytest -m integration`). "todo" rows are
properties the literature says matter that nothing here checks yet — the
first ones to add are #14/#15 (ablations of the two things that made #13
work) and #19 (capacity cliff), since the kernel port must preserve them.
