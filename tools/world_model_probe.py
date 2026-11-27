"""World-model probe: how well the trained state predicts the next frame's input
(evaluation only; same frames and targets for every checkpoint).

    python tools/world_model_probe.py --ref RUN --first 3000 --out probe.jsonl \
        base RUN_A/checkpoints/step_000689.pt RUN_B/checkpoints/...

Frames and actions come from one reference run (RUN/actor.jsonl, RUN/replay),
so a policy that paused its own game is scored on the same world as the rest.
Per decision t, in consecutive windows of --window decisions:
- x_t: the checkpoint's centre state (layers[12]) at the last action token of turn t;
- y_t: the untouched model's mean image-token input embedding of frame t (the
  target is fixed across checkpoints).
Ridge from x_t to y_{t+1}, fit on the first 70 % of pairs (alpha picked on the
last part of that), scored on the last 30 %: R^2 on per-dimension
standardized targets, averaged over dimensions. Baselines: copy (y_t as the
prediction of y_{t+1}) and ridge from y_t. "gain" = R^2(state) - R^2(ridge y_t):
what the state knows about the next frame beyond the current one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from gamescaling_trainer import CENTRE, assistant_content, read_rows  # noqa: E402

from sili_quartz.actor_objective import image_means, turn_positions  # noqa: E402
from sili_quartz.gamescaling import MINECRAFT_KEYS, MINECRAFT_RULES, USER_TEXT, build_system_prompt  # noqa: E402


def ridge_r2(xtr, ytr, xte, yte, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)) -> float:
    """Mean per-dimension R^2 of ridge x -> y on the test split (both standardized by train stats)."""
    def z(a, ref):
        return (a - ref.mean(0)) / (ref.std(0) + 1e-6)
    xte, yte = z(xte, xtr), z(yte, ytr)
    xtr, ytr = z(xtr, xtr), z(ytr, ytr)
    n = xtr.shape[0]
    cut = int(0.8 * n)

    def fit(x, y, alpha):
        d = x.shape[1]
        return torch.linalg.solve(x.T @ x + alpha * torch.eye(d, dtype=x.dtype), x.T @ y)

    def r2(w, x, y):
        return float(1 - ((x @ w - y) ** 2).mean(0).div(y.var(0) + 1e-9).mean())

    best = max(alphas, key=lambda al: r2(fit(xtr[:cut], ytr[:cut], al), xtr[cut:], ytr[cut:]))
    return r2(fit(xtr, ytr, best), xte, yte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="'base' or paths to trainer checkpoints (step_*.pt)")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--first", type=int, default=3000, help="use the reference run's first N decisions")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    rows = read_rows(a.ref / "actor.jsonl", a.ref / "replay")[:a.first]
    windows = [rows[i:i + a.window] for i in range(0, len(rows) - a.window + 1, a.window)]

    def inputs(win):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for r in win:
            img = Image.open(a.ref / "replay" / f"{r['n']:06d}.jpg").convert("RGB")
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": img}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content(r)}]})
        return proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt").to(a.device)

    targets: list[torch.Tensor] | None = None
    for ck in ["base"] + [c for c in a.checkpoints if c != "base"]:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
        if ck != "base":
            sd = torch.load(ck, map_location="cpu")
            missing = [k for k in sd if k not in dict(model.named_parameters())]
            model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        else:
            missing = []
        image_token = model.config.image_token_id
        xs, ys = [], []
        with torch.no_grad():
            for win in windows:
                inp = inputs(win)
                ids = inp["input_ids"][0].tolist()
                turns = turn_positions(ids, tok, prefill="<|action_start|")
                if len(turns) != len(win):
                    xs.append(None)
                    ys.append(None)
                    continue
                hs = model(**inp, output_hidden_states=True, logits_to_keep=1).hidden_states
                xs.append(hs[CENTRE][0, [t["action"][-1] for t in turns]].float().cpu())
                ys.append(image_means(ids, hs[0][0], image_token, [t["state"] for t in turns]).float().cpu())
        if targets is None:
            targets = ys  # the untouched model's inputs: the same target for every checkpoint
        X = torch.cat([x[:-1] for x, y in zip(xs, targets) if x is not None and y is not None]).double()
        Y = torch.cat([y[1:] for x, y in zip(xs, targets) if x is not None and y is not None]).double()
        Yc = torch.cat([y[:-1] for x, y in zip(xs, targets) if x is not None and y is not None]).double()
        n = X.shape[0]
        cut = int(0.7 * n)
        r2_state = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
        r2_cur = ridge_r2(Yc[:cut], Y[:cut], Yc[cut:], Y[cut:])
        yz = (Y[cut:] - Y[:cut].mean(0)) / (Y[:cut].std(0) + 1e-6)
        cz = (Yc[cut:] - Y[:cut].mean(0)) / (Y[:cut].std(0) + 1e-6)
        r2_copy = float(1 - ((cz - yz) ** 2).mean(0).div(yz.var(0) + 1e-9).mean())
        row = {"checkpoint": ck, "pairs": n, "test_pairs": n - cut, "r2_state": round(r2_state, 4),
               "r2_ridge_current_frame": round(r2_cur, 4), "r2_copy": round(r2_copy, 4),
               "gain": round(r2_state - r2_cur, 4), "unmatched_keys": len(missing)}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
