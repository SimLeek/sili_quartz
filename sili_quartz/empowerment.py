"""State empowerment through the input tether's world model (track A, rung R2).

Empowerment of a state is a channel capacity: how much the choice of action
can decide what comes next (Klyubin, Polani and Nehaniv 2005). Here:

1. From the state (the conversation up to the current frame and the action
   prefill), run K probe actions, each a short continuation on a copy of the
   prefix cache. The probes come from a fixed source (recent actions from the
   replay), not the acting policy, so the policy's own randomness earns
   nothing.
2. For each probe, read the input tether's prediction of the next frame:
   chunk 1's layer average at the probe's last token, standardized by the
   tether's bank.
3. Per dimension, Gaussian channel: 1/2 log(1 + spread across probes /
   sigma^2), with sigma^2 the tether's own prediction error per dimension.
   Summed over dimensions.

A static screen (pause, death, a dialog) gives the same prediction for
every probe, so it scores about 0. A poor world model has a large sigma^2
and so claims little control, which is why no separate gate is needed.
See docs/energy-controller/track-a-objective-math.md, sections 2 and 5.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from sili_quartz.gamescaling import PREFILL_DIRECT, USER_TEXT


def gaussian_capacity(preds: Tensor, sigma2: Tensor) -> Tensor:
    """preds: (K, d) standardized predictions, one row per probe action;
    sigma2: (d,) prediction error variance. Sum over d of 1/2 log(1 + spread / sigma2)."""
    spread = preds.var(0, unbiased=True)
    return 0.5 * torch.log1p(spread / sigma2.clamp(min=1e-6)).sum()


@torch.no_grad()
def state_empowerment(model, processor, system: str, history: list, frame, probes: list[str],
                      chunk: int, stats: dict) -> tuple[float, Tensor]:
    """history: [(frame, assistant content)] before the current frame; probes:
    raw action strings as the actor logs them (">... <|action_end|>"); stats:
    chunk-1 "mean_x", "std_x" (the tether's bank) and "sigma2". The model must
    be in eval mode (gradient checkpointing turns the cache off) and energy off.
    Returns (empowerment in nats, standardized predictions (K, d))."""
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for f, content in history:
        msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": f}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": frame}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": PREFILL_DIRECT}]})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                         continue_final_message=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], images=[f for f, _ in history] + [frame], return_tensors="pt").to(dev)
    pos, delta = model.model.get_rope_index(inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                                            image_grid_thw=inputs["image_grid_thw"])
    out = model(**inputs, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, n0 = out.past_key_values, inputs["input_ids"].shape[1]
    preds = []
    for raw in probes:
        ids = processor.tokenizer(raw, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        n = ids.shape[1]
        ppos = torch.arange(n0, n0 + n, device=dev).view(1, 1, n).expand(3, 1, n) + delta.to(dev)
        o = model(input_ids=ids, position_ids=ppos, past_key_values=copy.deepcopy(cache), use_cache=True,
                  output_hidden_states=True, logits_to_keep=1)
        hs = o.hidden_states
        preds.append(torch.stack([hs[k + 1][0, -1] for k in range(chunk)]).mean(0).float())
    p = (torch.stack(preds) - stats["mean_x"]) / stats["std_x"]
    return float(gaussian_capacity(p, stats["sigma2"])), p
