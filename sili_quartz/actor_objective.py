"""Track A objective parts (A.10): predictive information of the agent's own stream,
the health floor and the dual that sets the energy dose.

Stream features, per turn t of a replayed conversation (one turn = one frame
and the 200 ms action chunk chosen for it):
- s_t: the centre block's state (`layers[12]` output) at the end of the user
  turn, i.e. after seeing frame t and before acting;
- a_t: the mean `layers[12]` state over the action chunk's tokens.
Both are projected to d <= 64 by trainer-side linear maps (not part of the
model).

A1:      I(past; future) with past_t = [s_t, a_t], future_t = s_{t+1}.
A1-act:  past_t = s_t.
A2:      past_t = a_t (empowerment-like: actions to the next state).
A3:      past_t = region A (layers[6]) state, future_t = region B (layers[18])
         state at t+1.

The estimate is gaussian_bc (bias-corrected Gaussian, the benched primary)
over a bank of recent detached pairs plus the current window's pairs, which
carry the gradient (the representation term). The pointwise value
log p(f|p) - log p(f) under the same Gaussian fit is each chunk's reward for
the policy-gradient term.

Health: the participation ratio (PR) of the centre-block states over the
window's tokens. Floor = 0.8 x its value over the first steps. The dual
lambda <- max(0, lambda + eta * (floor - PR) / floor) is the energy dose
(1 = TRACK_A_COSTS, 0.03 x the P0c text dose; capped at 1.5) and weights the health penalty in the loss.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sili_quartz.energy_hooks import mean_abs_outputs
from sili_quartz.energy_scan import ScanEnergy, ScanSite

# --- energy sites (same as the P0c runs: MLP activation and attention output per layer) ---

P0C_COSTS = (0.03, 0.01)
# Track A's dose unit: 0.03 x P0c, the largest dose at which the model's free
# reasoning (thinking mode, 5-turn history) stays clean: no junk characters and
# every action parses at <= 0.03 x; first junk at 0.05 x; 5/15 parse at 0.1 x.
# Actions alone (restricted vocabulary) still parse at 0.1 x, which hid the damage.
TRACK_A_COSTS = (0.0009, 0.0003)
DECAY_DIV = 40.0


def site_modules(model) -> tuple[list[nn.Module], list[nn.Module]]:
    layers = model.model.language_model.layers
    mlp = [layer.mlp.act_fn for layer in layers]
    attn = [getattr(layer, "linear_attn", None) or layer.self_attn for layer in layers]
    return mlp, attn


@torch.no_grad()
def measure_site_means(model, run) -> tuple[list[float], list[float]]:
    mlp, attn = site_modules(model)
    means = mean_abs_outputs(mlp + attn, run)
    return means[: len(mlp)], means[len(mlp):]


def io_modules(model) -> list[nn.Module]:
    """Input neurons: the token embeddings and the vision merger's output (image tokens)."""
    return [model.model.language_model.embed_tokens, model.model.visual.merger]


@torch.no_grad()
def measure_io_means(model, run) -> list[float]:
    return mean_abs_outputs(io_modules(model), run)


def choice_columns(tok, vocab: list[int]) -> list[int]:
    """Columns of the action vocabulary that choose something (digits, '-', key
    names), not format: separators, spaces, the end tag and end of turn. Energy
    drains always-on neurons, so format tokens would be shut off and actions
    would stop parsing."""
    cols = []
    for j, t in enumerate(vocab):
        text = tok.decode([t], skip_special_tokens=False)
        core = text.strip()
        if not core or core == ";" or "<" in text or ">" in text or t == tok.eos_token_id:
            continue
        cols.append(j)
    return cols


def make_energy(model, mlp_means, attn_means, costs=TRACK_A_COSTS, decay_div=DECAY_DIV,
                io_means=None, n_choice: int = 0) -> ScanEnergy:
    """Every site is scaled to its layer's MLP activations, so a cost is the same dose everywhere.
    io_means: also the input neurons (io_modules). n_choice > 0: also an output site over
    the action-choice probabilities (`output_energy`), scaled so a uniform spread is 1."""
    mlp, attn = site_modules(model)
    sites = [ScanSite(m, ref, 1.0, costs[0]) for m, ref in zip(mlp, mlp_means)]
    sites += [ScanSite(m, ref, mean / ref, costs[1]) for m, mean, ref in zip(attn, attn_means, mlp_means)]
    ref = sum(mlp_means) / len(mlp_means)
    if io_means is not None:
        emb, merger = io_modules(model)
        sites += [ScanSite(emb, ref, io_means[0] / ref, costs[1]),
                  ScanSite(merger, ref, io_means[1] / ref, costs[1], flat_tokens=True)]
    out_index = None
    if n_choice:
        out_index = len(sites)
        sites.append(ScanSite(nn.Identity(), 1.0, 1.0 / n_choice, costs[0]))
    energy = ScanEnergy(sites, decay_div=decay_div)
    energy.out_index = out_index
    return energy


def output_energy(energy: ScanEnergy, logits: Tensor, cols: Tensor) -> Tensor:
    """Energy on the output neurons: the probabilities of the action-choice columns
    `cols` of `logits` (..., V) over the action vocabulary, as one sequence over the
    leading dims. Kept probabilities pass (with gradient), fired ones are set to
    2 / n_choice, shut-off ones to near 0; returns logits on the same normalizer."""
    if getattr(energy, "out_index", None) is None or not energy.enabled:
        return logits
    shape = logits.shape
    flat = logits.reshape(-1, shape[-1]).float()
    lse = flat.logsumexp(-1, keepdim=True)
    p = (flat - lse).exp()[:, cols]
    p2 = energy.apply(energy.out_index, p[None])[0]
    out = flat.clone()
    out[:, cols] = p2.clamp_min(1e-12).log() + lse
    return out.reshape(shape).to(logits.dtype)


def decoder_attn_post(energy: ScanEnergy, n_layers: int):
    """For GraphDecoder.attn_post: energy on layer i's attention output (site n_layers + i)."""
    def post(i, x):
        return energy.apply(n_layers + i, x) if energy.enabled else x
    return post


# --- Gaussian predictive information ---------------------------------------------------------

def _logdet_cov(x: Tensor) -> Tensor:
    xc = x - x.mean(0, keepdim=True)
    cov = xc.T @ xc / (x.shape[0] - 1)
    cov = cov + 1e-5 * torch.eye(cov.shape[0], device=x.device, dtype=x.dtype)
    return torch.linalg.slogdet(cov)[1]


def wishart_bias(d: int, n: int) -> float:
    i = torch.arange(1, d + 1, dtype=torch.float64)
    return float(torch.special.digamma((n - i) / 2).sum() + d * math.log(2 / (n - 1)))


def gaussian_bc(x: Tensor, y: Tensor) -> Tensor:
    """Bias-corrected Gaussian MI in nats; differentiable in x and y."""
    n, dx, dy = x.shape[0], x.shape[1], y.shape[1]
    raw = 0.5 * (_logdet_cov(x) + _logdet_cov(y) - _logdet_cov(torch.cat([x, y], 1)))
    return raw - 0.5 * (wishart_bias(dx, n) + wishart_bias(dy, n) - wishart_bias(dx + dy, n))


@torch.no_grad()
def pointwise_mi(xb: Tensor, yb: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """log p(y|x) - log p(y) for the rows of (x, y), under the Gaussian fitted to (xb, yb)."""
    xy = torch.cat([xb, yb], 1).double()
    mu = xy.mean(0)
    cov = torch.cov(xy.T) + 1e-5 * torch.eye(xy.shape[1], dtype=torch.float64, device=xy.device)
    dx = xb.shape[1]
    sxx, sxy, syy = cov[:dx, :dx], cov[:dx, dx:], cov[dx:, dx:]
    w = torch.linalg.solve(sxx, sxy)  # (dx, dy)
    s_res = syy - sxy.T @ w
    xc, yc = x.double() - mu[:dx], y.double() - mu[dx:]
    r_cond = yc - xc @ w

    def logn(r, s):
        L = torch.linalg.cholesky(s)
        z = torch.cholesky_solve(r.T, L).T
        return -0.5 * ((r * z).sum(1) + 2 * torch.log(torch.diagonal(L)).sum())

    return (logn(r_cond, s_res) - logn(yc, syy)).float()


class PredictiveInfo(nn.Module):
    """Projections, the pair bank and the estimate for one arm. The bank keeps
    raw (d_model) features of recent pairs and is re-projected every step, so
    it never goes stale as the projections train."""

    PAST = {"A1": ("s", "a"), "A1-act": ("s",), "A1+self": ("s", "a"), "A1-lam": ("s", "a"),
            "A1+bored": ("s", "a"), "A2": ("a",), "A3": ("rA",)}

    def __init__(self, arm: str = "A1", d_model: int = 1024, d: int = 32, bank: int = 1024):
        super().__init__()
        self.arm = arm
        self.past_parts = self.PAST[arm]
        self.fut_key = "rB" if arm == "A3" else "s"
        self.proj = nn.ModuleDict({k: nn.Linear(d_model, d) for k in ("s", "a", "rA", "rB")})
        self.self_head = nn.Linear(d, d) if arm == "A1+self" else None
        self.d, self.bank_size = d, bank
        self.bank: dict[str, Tensor] | None = None  # raw past parts + "future"
        self.noise_seed = 0

    def ready(self) -> bool:
        """Enough banked pairs for a covariance in the joint dimension."""
        dx = self.d * len(self.past_parts)
        return self.bank is not None and self.bank["future"].shape[0] >= 2 * (dx + self.d)

    @property
    def bank_rows(self) -> int:
        return 0 if self.bank is None else self.bank["future"].shape[0]

    def _project(self, raw: dict[str, Tensor], ref: dict[str, Tensor] | None = None,
                 noise: bool = True) -> tuple[Tensor, Tensor]:
        """Project, standardize each dimension by the bank's statistics (ref), and add
        unit Gaussian noise: the estimate is then the MI of a noisy channel, at most
        1/2 log 2 nats per dimension, so a static (near-deterministic) code cannot
        drive it to infinity."""
        ref = raw if ref is None else ref

        def code(k_list, r_src, r_ref, stream):
            z = torch.cat([self.proj[k](r_src[k].float()) for k in k_list], 1)
            zr = torch.cat([self.proj[k](r_ref[k].float()) for k in k_list], 1)
            z = (z - zr.mean(0)) / (zr.std(0) + 1e-5)
            if noise:
                g = torch.Generator(device=z.device).manual_seed(2 * self.noise_seed + stream)
                z = z + torch.randn(z.shape, generator=g, device=z.device)
            return z

        x = code(self.past_parts, raw, ref, 0)
        fut = {self.fut_key: raw["future"]}
        fut_ref = {self.fut_key: ref["future"]}
        y = code((self.fut_key,), fut, fut_ref, 1)
        return x, y

    def forward(self, feats: dict[str, Tensor]) -> dict:
        """feats: name -> (T, d_model) for consecutive turns."""
        raw = {k: feats[k][:-1] for k in self.past_parts}
        raw["future"] = feats[self.fut_key][1:]
        out = {"pairs": raw["future"].shape[0]}
        self.noise_seed += 1
        if not self.ready():
            out["mi"] = None
            out["reward"] = torch.zeros(out["pairs"], device=raw["future"].device)
        else:
            both = {k: torch.cat([self.bank[k], raw[k]], 0) for k in raw}
            xa, ya = self._project(both)
            nb = self.bank["future"].shape[0]
            xb, yb, x, y = xa[:nb], ya[:nb], xa[nb:], ya[nb:]
            out["mi"] = gaussian_bc(xa, ya)
            out["reward"] = pointwise_mi(xb.detach(), yb.detach(), x.detach(), y.detach())
        if self.self_head is not None:
            a = self.proj["a"](feats["a"].float())
            s_ = self.proj["s"](feats["s"].float())
            out["self_loss"] = (self.self_head(s_[:-1]) - a[1:].detach()).pow(2).mean()
        self._push({k: v.detach() for k, v in raw.items()})
        return out

    def _push(self, raw: dict[str, Tensor]) -> None:
        if self.bank is None:
            self.bank = raw
        else:
            self.bank = {k: torch.cat([self.bank[k], v], 0)[-self.bank_size:] for k, v in raw.items()}


# --- health and the dual ------------------------------------------------------------------------

def participation_ratio(h: Tensor) -> Tensor:
    """(sum lambda)^2 / sum lambda^2 of the covariance of h's rows (tokens); differentiable."""
    hc = h.float() - h.float().mean(0, keepdim=True)
    g = hc @ hc.T  # tokens x tokens: same nonzero eigenvalues, cheaper when tokens < width
    return g.diagonal().sum() ** 2 / (g * g).sum()


class Dual:
    """lambda <- max(0, lambda + eta * (floor - health) / floor); floor = frac x early health."""

    def __init__(self, frac: float = 0.8, eta: float = 0.05, warmup: int = 20, lam_max: float = 1.5,
                 enabled: bool = True):
        self.frac, self.eta, self.warmup, self.lam_max, self.enabled = frac, eta, warmup, lam_max, enabled
        self.early: list[float] = []
        self.floor: float | None = None
        self.lam = 0.0

    def update(self, health: float) -> float:
        if self.floor is None:
            self.early.append(health)
            if len(self.early) >= self.warmup:
                self.floor = self.frac * sum(self.early) / len(self.early)
            return self.lam
        if self.enabled:
            self.lam = min(self.lam_max, max(0.0, self.lam + self.eta * (self.floor - health) / self.floor))
        return self.lam


class Boredom:
    """A1+bored: raise the dose while the short-run information gain is below its long-run average."""

    def __init__(self, short: float = 0.1, long: float = 0.01, step: float = 0.05, lam_max: float = 1.5):
        self.short = self.long = None
        self.a_s, self.a_l, self.step, self.lam_max = short, long, step, lam_max
        self.extra = 0.0

    def update(self, gain: float) -> float:
        self.short = gain if self.short is None else (1 - self.a_s) * self.short + self.a_s * gain
        self.long = gain if self.long is None else (1 - self.a_l) * self.long + self.a_l * gain
        self.extra = min(self.lam_max, max(0.0, self.extra + (self.step if self.short < self.long else -self.step)))
        return self.extra


# --- where the turns are in a rendered conversation -------------------------------------------

def turn_positions(ids: list[int], tok, prefill: str = "<think>\n\n</think>\n\n<|action_start|") -> list[dict]:
    """For each (user, assistant) pair: 'state' = index of the user turn's last
    token (<|im_end|>), 'action' = indices of the sampled action tokens (after
    the fixed prefill, up to the assistant's <|im_end|>)."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    starts = [i for i, t in enumerate(ids) if t == im_start]
    turns, state = [], None
    for si, s in enumerate(starts):
        role = tok.decode(ids[s + 1:s + 2]).strip()
        end = next(j for j in range(s + 1, len(ids)) if ids[j] == im_end) if im_end in ids[s + 1:] else len(ids)
        if role == "user":
            state = end
        elif role == "assistant" and state is not None:
            body = ids[s + 1:end]
            first = None
            for j in range(1, len(body) + 1):
                if tok.decode(body[:j]).endswith(prefill):
                    first = j
                    break
            if first is not None:
                turns.append({"state": state, "action": list(range(s + 1 + first, end))})
            state = None
    return turns
