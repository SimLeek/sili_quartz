"""Mutual-information estimators: speed and accuracy against known MI. GPU.

Data: x ~ N(0, I_d); y_i = rho*x_i + sqrt(1-rho^2)*n_i, so true MI = -d/2*log(1-rho^2).
"cubic": y_i -> y_i**3. MI is unchanged by an invertible per-dim map, but the
Gaussian estimators no longer see it.
Reference cost: one forward+backward of a 2-layer MLP (width 256) on the same batch.
"""

import json
import math
import sys
import time

import torch
from torch import nn

dev = "cuda:0"
torch.manual_seed(0)
N = 1024  # one window of samples
STEPS = 300  # training steps for the critic-based estimators


def data(d, rho, cubic, n=N, gen=None):
    x = torch.randn(n, d, device=dev, generator=gen)
    y = rho * x + math.sqrt(1 - rho**2) * torch.randn(n, d, device=dev, generator=gen)
    return x, (y**3 if cubic else y)


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / reps


def logdet_cov(z):
    z = z - z.mean(0)
    c = z.T @ z / (len(z) - 1) + 1e-6 * torch.eye(z.shape[1], device=dev)
    return torch.linalg.slogdet(c)[1]


def gaussian(x, y):
    return 0.5 * (logdet_cov(x) + logdet_cov(y) - logdet_cov(torch.cat([x, y], 1))).item()


def linear_predictive(x, y):
    """Der-style: fit y ~ A x by least squares, MI = 1/2 (logdet Sy - logdet S_resid)."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    a = torch.linalg.lstsq(xc, yc).solution
    return 0.5 * (logdet_cov(y) - logdet_cov(yc - xc @ a)).item()


def ksg(x, y, k=3):
    """Kraskov-Stoegbauer-Grassberger estimator 1, max-norm, O(N^2 d)."""
    n = len(x)
    dx = torch.cdist(x, x, p=float("inf"))
    dy = torch.cdist(y, y, p=float("inf"))
    dz = torch.maximum(dx, dy)
    dz.fill_diagonal_(float("inf"))
    eps = dz.kthvalue(k, 1).values[:, None]
    nx = (dx < eps).sum(1) - 1
    ny = (dy < eps).sum(1) - 1
    dig = torch.special.digamma
    return (dig(torch.tensor(float(k))) + dig(torch.tensor(float(n))) - (dig(nx + 1.0) + dig(ny + 1.0)).mean()).item()


def mlp(d_in, d_out, w=256):
    return nn.Sequential(nn.Linear(d_in, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, d_out)).to(dev)


def critic_estimator(kind, d, rho, cubic):
    """InfoNCE (separable critic) or MINE (DV bound, joint critic), trained on fresh batches."""
    gen = torch.Generator(dev).manual_seed(1)
    if kind == "infonce":
        f, g = mlp(d, 64), mlp(d, 64)
        params = [*f.parameters(), *g.parameters()]
    else:
        t_net = mlp(2 * d, 1)
        params = list(t_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def bound(x, y):
        if kind == "infonce":
            s = f(x) @ g(y).T
            return math.log(len(x)) - nn.functional.cross_entropy(s, torch.arange(len(x), device=dev))
        joint = t_net(torch.cat([x, y], 1)).mean()
        marg = t_net(torch.cat([x, y[torch.randperm(len(y), device=dev, generator=gen)]], 1)).squeeze(1)
        return joint - (torch.logsumexp(marg, 0) - math.log(len(marg)))

    def step():
        x, y = data(d, rho, cubic, gen=gen)
        loss = -bound(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    total = time.perf_counter() - t
    with torch.no_grad():
        est = sum(bound(*data(d, rho, cubic, gen=gen)).item() for _ in range(10)) / 10
        _, infer = timed(lambda: bound(*data(d, rho, cubic, gen=gen)))
    return est, total / STEPS, total, infer


def reference_nn(d):
    net, x = mlp(d, d), torch.randn(N, d, device=dev)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        net(x).square().mean().backward()
        opt.step()

    return timed(step, 20)[1]


print(json.dumps({"device": torch.cuda.get_device_name(0), "N": N, "critic_steps": STEPS}), flush=True)
for d in (16, 64, 256):
    ref = reference_nn(d)
    for target in (2.0, 8.0):
        rho = math.sqrt(1 - math.exp(-2 * target / d))
        for cubic in (False, True):
            row = {"d": d, "true_mi": target, "cubic": cubic, "ref_nn_step_ms": round(ref * 1e3, 3)}
            x, y = data(d, rho, cubic)
            for name, fn in (("gaussian", gaussian), ("linear_pred", linear_predictive), ("ksg", ksg)):
                est, sec = timed(lambda fn=fn: fn(x, y))
                row[name] = [round(est, 2), round(sec * 1e3, 3)]  # [estimate, ms per call]
            for kind in ("infonce", "mine"):
                est, per_step, total, infer = critic_estimator(kind, d, rho, cubic)
                row[kind] = [round(est, 2), round(per_step * 1e3, 3), round(total, 2), round(infer * 1e3, 3)]  # [est, ms/step, s to train, ms/eval]
            print(json.dumps(row), flush=True)
print("DONE", flush=True)
sys.exit(0)
