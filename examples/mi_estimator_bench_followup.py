"""Follow-up: bias-corrected Gaussian MI, and InfoNCE trained 10x longer. GPU."""
import json, math, sys, time
sys.argv = [sys.argv[0]]
from pathlib import Path
exec(Path(__file__).with_name("mi_estimator_bench.py").read_text().split('print(json.dumps({"device"')[0])

def wishart_bias(d, n):
    """E[log det S] - log det Sigma for a sample covariance with n-1 degrees of freedom."""
    i = torch.arange(1, d + 1, dtype=torch.float64)
    return (torch.special.digamma((n - i) / 2).sum() + d * math.log(2 / (n - 1))).item()

def gaussian_bc(x, y):
    n, dx, dy = len(x), x.shape[1], y.shape[1]
    raw = 0.5 * (logdet_cov(x) + logdet_cov(y) - logdet_cov(torch.cat([x, y], 1))).item()
    return raw - 0.5 * (wishart_bias(dx, n) + wishart_bias(dy, n) - wishart_bias(dx + dy, n))

STEPS = 3000
for d in (16, 64, 256):
    for target in (2.0, 8.0):
        rho = math.sqrt(1 - math.exp(-2 * target / d))
        for cubic in (False, True):
            x, y = data(d, rho, cubic)
            est, sec = timed(lambda: gaussian_bc(x, y))
            row = {"d": d, "true_mi": target, "cubic": cubic, "gaussian_bc": [round(est, 2), round(sec * 1e3, 3)]}
            x4, y4 = data(d, rho, cubic, n=4 * N)
            est4, sec4 = timed(lambda: gaussian_bc(x4, y4))
            row["gaussian_bc_N4096"] = [round(est4, 2), round(sec4 * 1e3, 3)]
            if d >= 64:
                e, per, tot, inf = critic_estimator("infonce", d, rho, cubic)
                row["infonce_3000"] = [round(e, 2), round(per * 1e3, 3), round(tot, 2), round(inf * 1e3, 3)]
            print(json.dumps(row), flush=True)
print("DONE", flush=True)
