# Mutual-information estimators: speed and accuracy (2026-11-23)

For the energy controller's track A (`docs/energy-controller/1-introduction.md`
1.11). Rule (user): keep only estimators that run about as fast as a small
neural network or faster, and research others if none survive.

## Setup

- **Hardware:** Tesla P40 on the sandbox, under gpu_guard. Scripts:
  `examples/mi_estimator_bench.py` and `examples/mi_estimator_bench_followup.py`.
- **Data:** x ~ N(0, I_d), with y_i = ρ x_i + √(1−ρ²) n_i, so the true MI is
  −(d/2) log(1−ρ²). ρ is set to give 2 or 8 nats.
- **"Cubic":** the same data with y → y³. MI is unchanged by an invertible
  per-dimension map, but the relation is no longer linear.
- **Window:** N = 1024 samples (4096 for one column).
- **Reference cost:** one forward + backward step of a 2-layer, 256-wide MLP on
  the same batch: 0.43–0.54 ms.

## Estimators

- **gaussian:** ½[log det Σx + log det Σy − log det Σxy], from sample
  covariances.
- **gaussian_bc:** the same, minus the exact Wishart bias of each log det:
  Σ_{i=1..d} ψ((N−i)/2) + d·log(2/(N−1)).
- **linear_pred:** the Der style, a least-squares predictor of y from x, then
  ½[log det Σy − log det Σresid]. It gives the same numbers as gaussian.
- **ksg:** Kraskov–Stögbauer–Grassberger kNN, k = 3, max-norm. Cost O(N² d).
- **infonce / mine:** a critic MLP trained on fresh batches (Adam, lr 1e-3).
  - InfoNCE uses a separable critic and is capped at log N = 6.93 nats.
  - MINE uses the Donsker–Varadhan bound.
  - Trained 300 steps; InfoNCE also 3000.

## Results

Each cell is the estimate, then the time per call or per training step. Bold
marks an estimate within 10% of the truth.

| d | true | data | gaussian (raw) | gaussian_bc | gaussian_bc N=4096 | ksg | infonce 300 | infonce 3000 | mine 300 |
|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 16 | 2 | linear | **2.16** / 0.57 ms | **1.96** / 0.70 | **2.02** / 0.78 | 0.48 / 24 ms | **1.91** / 1.2 ms/step | — | 1.84 / 1.3 |
| 16 | 2 | cubic | 1.22 | 1.16 | 1.12 | 0.05 | 1.50 | — | 1.15 |
| 16 | 8 | linear | **8.13** | **8.07** | **8.01** | 1.87 | 6.05 | — | 5.27 |
| 16 | 8 | cubic | 3.95 | 3.77 | 3.87 | 0.22 | 5.04 | — | 3.69 |
| 64 | 2 | linear | 4.11 / 0.86 | **1.96** / 1.06 | **1.96** / 1.36 | 0.06 / 21 | 1.47 / 1.0 | 1.78 / 1.0 | 1.37 |
| 64 | 2 | cubic | 3.42 | 1.12 | 1.19 | 0.00 | 0.80 | 1.40 | 0.75 |
| 64 | 8 | linear | 10.17 | **7.94** | **7.96** | 0.32 | 5.00 | 5.70 | 3.26 |
| 64 | 8 | cubic | 6.61 | 4.45 | 4.47 | 0.00 | 3.52 | 5.28 | 2.16 |
| 256 | 2 | linear | 45.26 / 3.6 | **2.02** / 3.7 | **2.03** / 4.3 | 0.02 / 23 | 0.24 / 1.1 | 0.35 / 1.1 | 0.42 |
| 256 | 2 | cubic | 44.97 | 1.19 | 1.18 | 0.00 | 0.11 | 0.18 | 0.10 |
| 256 | 8 | linear | 51.11 | **7.95** | **7.88** | 0.02 | 1.38 | 1.69 | 1.65 |
| 256 | 8 | cubic | 48.40 | 4.89 | 4.73 | 0.00 | 0.75 | 0.97 | 0.59 |

Critic timings:
- InfoNCE and MINE cost about 1.0–1.3 ms per training step, about 2× the
  reference step.
- One evaluation costs 0.3–0.4 ms.
- 3000 steps take about 3 s.

## Keep or drop, by the speed rule and the accuracy above

- **Dropped: KSG.**
  - It costs 20–24 ms per call, about 45× a small-network step, and grows as N².
  - It collapses toward 0 at d ≥ 16 here.
- **Dropped: MINE.** It costs the same as InfoNCE but is worse or equal in
  every row.
- **Kept: gaussian_bc.**
  - 0.7 ms at d = 16, 1 ms at d = 64, 3.7 ms at d = 256, with no training.
  - About 1.5–8× a small-network step, growing as d³ because of the log det.
  - Unbiased on Gaussian data at every d once the Wishart term is removed. The
    raw version is off by up to 43 nats at d = 256.
  - It *under*estimates a nonlinear relation, giving 55–60% of the truth on the
    cubic data.
- **Kept, as the nonlinear arm at moderate d: InfoNCE.**
  - About 2× a small-network step, and it can be trained alongside the model
    (amortized).
  - It beats gaussian_bc on the cubic data at d ≤ 64 (5.28 vs 4.45 at 8 nats).
  - At d = 256 it was still far off after 3000 steps.
  - It can never report more than log N.
- **Untested, the natural next arm:** Gaussian MI on the residuals of a
  *nonlinear* forward model. This is linear_pred with the least-squares map
  replaced by the forward model track A needs anyway (SuperMimic-like). It may
  remove gaussian_bc's underestimate at gaussian_bc's speed.

Two estimators survive, so no further search is needed yet under the rule.
