"""Where does GameScaling-0.8B's training memory go? P40, fp32, text path, batch 1.

Peak memory of one forward+backward at several lengths, plus the bytes autograd
saves for backward, attributed to the innermost module that saved them.
"""

import json
import os
from collections import defaultdict

import torch
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM

dev = "cuda:0"
MODEL = os.path.expanduser("~/models/GameScaling-0.8B-work")
model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL, config=Qwen3_5Config.from_pretrained(MODEL).text_config, dtype=torch.float32
).to(dev)
model.train()
for p in model.parameters():
    p.requires_grad_(True)

stack = []
saved = defaultdict(int)


def pre(name):
    def hook(_m, _a):
        stack.append(name)

    return hook


def post(_m, _a, _o):
    stack.pop()


for name, m in model.named_modules():
    kind = type(m).__name__
    if name.endswith("linear_attn") or name.endswith("self_attn") or name.endswith("mlp") or name == "lm_head":
        m.register_forward_pre_hook(pre(kind if name != "lm_head" else "lm_head"))
        m.register_forward_hook(post)


def pack(t):
    if isinstance(t, torch.Tensor) and t.device.type == "cuda":
        saved[stack[-1] if stack else "other"] += t.numel() * t.element_size()
    return t


weights = torch.cuda.memory_allocated() / 2**20
print(json.dumps({"weights_mib": round(weights)}), flush=True)
for T in (256, 512, 1024):
    saved.clear()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 200000, (1, T), device=dev)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits = model(input_ids=ids, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    after_fwd = torch.cuda.memory_allocated() / 2**20
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**20
    row = {
        "T": T,
        "after_forward_mib": round(after_fwd - weights),
        "peak_mib": round(peak - weights),
        "saved_mib": {k: round(v / 2**20) for k, v in sorted(saved.items(), key=lambda kv: -kv[1])},
        "saved_kib_per_token": {k: round(v / 2**10 / T) for k, v in sorted(saved.items(), key=lambda kv: -kv[1])},
    }
    print(json.dumps(row), flush=True)
    del logits, loss
print("DONE", flush=True)
