"""Asynchronous trainer for the live GameScaling actor (track A, A.9-A.11).

Runs beside `gamescaling_actor.py --replay-dir RUN/replay --sync-dir RUN/sync`
on another GPU (the P40). Each step:
1. picks a window of `--window` consecutive decisions from the last
   `--recent` decisions of the actor's log (frames from RUN/replay);
2. renders them as the actor's conversation (system; user frame, assistant
   action, ...), runs the fp32 model with continual LoRA on every Linear and
   direct training of every other parameter (sili_quartz/continual.py), with
   energy at the actor's dose (sili_quartz/energy_scan.py);
3. loss = - w_rep * I(past; future)          (representation term, gaussian_bc)
          + w_pg * PPO clipped surrogate        (behaviour term: per-token ratio
                                               to the actor's sampling log-prob,
                                               clipped to 1 +- eps; reward = the
                                               pointwise MI, normalized)
          + lambda * relu(floor - PR) / floor (health; PR of the centre block)
          [+ lambda_act * relu(floor_act - H) / floor_act with --health pr+act:
           H = entropy of the action tokens; dose = max(lambda, lambda_act)]
          [+ the self-prediction loss for A1+self];
   log pi is over the actor's own restricted action vocabulary;
4. every `--merge-every` steps: merge the adapters into the weights, write a
   sync packet (exact fp16 diffs) and the dose for the actor (dose.json).

Arms (--arm): A1, A1-act, A1+self, A1-lam, A1+bored, A2, A3.
Dose (--dose): dual (lambda, the default), fixed:<x>, none.
The replay is recent play from slightly older weights; the PPO ratio is
taken against the log-prob the actor actually sampled with (logged per
token), so the clip bounds how far each update moves from the policy that
played.

    python examples/gamescaling_trainer.py --run ~/runs/a1_x --arm A1 --seconds 7200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

from sili_quartz.actor_objective import (
    Boredom, Dual, PredictiveInfo, choice_columns, make_energy, measure_io_means, measure_site_means,
    output_energy, participation_ratio, turn_positions,
)
from sili_quartz.continual import (
    ContinualLoRA, WeightSync, direct_params, reset_adam_state, sync_tensors, target_linears,
)
from sili_quartz.gamescaling import (
    ACTION_END, MINECRAFT_KEYS, MINECRAFT_RULES, NOOP_TEXT, USER_TEXT, ActionGrammar, action_prefix_state,
    action_vocab, masked_log_probs, build_system_prompt, reasoning_probe,
)

CENTRE, REGION_A, REGION_B = 13, 7, 19  # hidden_states index = layer index + 1


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def assistant_content(row: dict) -> str:
    raw = row.get("raw") or ""
    if row.get("parsed") and raw.rstrip().endswith(ACTION_END):
        return "<|action_start|" + raw
    return NOOP_TEXT


def turn_mask(grammar, tgt: list[int]):
    """The grammar mask at each position of one turn's action tokens (tgt:
    their action-vocab indices, -1 outside it), as the actor applied it;
    positions after the action closed (the filled-in end tag) are unmasked."""
    import torch

    rows, gtext = [], ""
    for j, i in enumerate(tgt):
        if action_prefix_state(gtext) == "done":
            rows.append([True] * len(grammar.strings))
        else:
            rows.append(grammar.allowed(gtext, j))
        gtext += grammar.strings[i] if i >= 0 else ""
    return torch.tensor(rows, dtype=torch.bool)


def read_rows(log: Path, replay: Path) -> list[dict]:
    rows = []
    if not log.exists():
        return rows  # the actor has not logged its first decision yet
    with open(log) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break  # a line still being written
            if (replay / f"{r['n']:06d}.jpg").exists():
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arm", default="A1", choices=list(PredictiveInfo.PAST))
    ap.add_argument("--dose", default="dual", help="dual | fixed:<x> | none")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--recent", type=int, default=600, help="sample windows from the last N decisions")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5, help="LoRA adapters")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm clip")
    ap.add_argument("--ppo-eps", type=float, default=0.2, help="PPO ratio clip")
    ap.add_argument("--checkpoint-every", type=float, default=1800.0, help="seconds between full checkpoints")
    ap.add_argument("--lr-direct", type=float, default=1e-5, help="norms, gates, biases, ...")
    ap.add_argument("--w-rep", type=float, default=1.0)
    ap.add_argument("--w-pg", type=float, default=1.0)
    ap.add_argument("--w-health", type=float, default=1.0)
    ap.add_argument("--energy-io", action="store_true",
                    help="energy also on the input neurons (token embeddings, vision merger) and the output "
                         "neurons (action-choice probabilities); the actor must run with --energy-io too")
    ap.add_argument("--health", choices=("pr", "pr+act"), default="pr",
                    help="pr+act: also hold the action entropy above 0.8 x its early value (second dual)")
    ap.add_argument("--grammar", action="store_true",
                    help="the actor sampled under the action-format mask: apply the same mask to log pi and "
                         "the entropy")
    ap.add_argument("--merge-every", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-every", type=float, default=300.0,
                    help="seconds between reasoning probes of the trained weights (0 = off)")
    ap.add_argument("--probe-samples", type=int, default=6)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    run, replay = a.run, a.run / "replay"
    log_path = run / "trainer.jsonl"
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    vocab = torch.tensor(action_vocab(tok), device=a.device)
    vocab_index = torch.full((len(tok),), -1, dtype=torch.long, device=a.device)
    vocab_index[vocab] = torch.arange(len(vocab), device=a.device)
    grammar = ActionGrammar(tok, vocab.tolist()) if a.grammar else None
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)

    linears = target_linears(model)
    lora = ContinualLoRA(linears, rank=a.rank, seed=a.seed)
    direct = direct_params(model, linears)
    for p in direct.values():
        p.requires_grad_(True)
    pi = PredictiveInfo(a.arm).to(a.device)
    opt = torch.optim.AdamW([
        {"params": list(lora.parameters()), "lr": a.lr},
        {"params": list(direct.values()), "lr": a.lr_direct, "weight_decay": 0.0},
        {"params": list(pi.parameters()), "lr": 1e-3},
    ])
    sync = WeightSync(sync_tensors(linears, direct), run / "sync")
    dual = Dual(enabled=a.arm != "A1-lam")
    bored = Boredom() if a.arm == "A1+bored" else None
    dual_act = Dual(enabled=a.health == "pr+act" and a.arm != "A1-lam")

    def dose_now() -> float:
        if a.dose == "none" or a.arm == "A1-lam":
            return 0.0
        if a.dose.startswith("fixed:"):
            return float(a.dose.split(":")[1])
        return max(dual.lam, dual_act.lam) + (bored.extra if bored else 0.0)

    def window_inputs(rows: list[dict]):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for r in rows:
            img = Image.open(replay / f"{r['n']:06d}.jpg").convert("RGB")
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": img}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content(r)}]})
        return proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt").to(a.device)

    # wait for the actor's first windows
    while len(read_rows(run / "actor.jsonl", replay)) < a.window + 1:
        time.sleep(2)
    # the actor measures the energy site means; use the same ones so a dose means the same
    means_path = run / "energy_means.json"
    for _ in range(60):
        if means_path.exists():
            break
        time.sleep(2)
    if means_path.exists():
        m = json.loads(means_path.read_text())
        mlp_means, attn_means, io_means = m["mlp"], m["attn"], m.get("io")
    else:
        rows = read_rows(run / "actor.jsonl", replay)
        first = window_inputs(rows[-a.window:])
        with torch.no_grad():
            mlp_means, attn_means = measure_site_means(model, lambda: model(**first, logits_to_keep=1))
            io_means = measure_io_means(model, lambda: model(**first, logits_to_keep=1)) if a.energy_io else None
    cols = choice_columns(tok, vocab.tolist()) if a.energy_io else []
    cols_t = torch.tensor(cols, device=a.device)
    energy = make_energy(model, mlp_means, attn_means, io_means=io_means if a.energy_io else None,
                         n_choice=len(cols))
    adv_mean, adv_var, step, t_start = 0.0, 1.0, 0, time.time()
    last_ckpt = t_start
    last_probe = 0.0

    def probe() -> None:
        """What the model being trained thinks on the newest frame (evaluation only;
        RUN/probes.jsonl). No history: with past turns the model copies their empty
        thinking blocks and almost never thinks."""
        rows_ = read_rows(run / "actor.jsonl", replay)[-6:]
        if len(rows_) < 6:
            return
        hist = [(Image.open(replay / f"{r['n']:06d}.jpg").convert("RGB"), assistant_content(r)) for r in rows_[:-1]]
        frame = Image.open(replay / f"{rows_[-1]['n']:06d}.jpg").convert("RGB")
        d0 = dose_now()
        energy.enabled = d0 > 0
        energy.set_dose(cost_scale=d0, noise_scale=d0)
        for k in range(a.probe_samples):
            res = reasoning_probe(model, proc, system, frame)
            with open(run / "probes.jsonl", "a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 1), "step": step, "decision": rows_[-1]["n"],
                                     "sample": k, "dose": d0, **res}) + "\n")
    print(json.dumps({"arm": a.arm, "dose": a.dose, "lora_params": sum(p.numel() for p in lora.parameters()),
                      "direct_params": sum(p.numel() for p in direct.values())}), flush=True)

    while (not a.seconds or time.time() - t_start < a.seconds) and (not a.steps or step < a.steps):
        if a.probe_every and time.time() - last_probe > a.probe_every:
            last_probe = time.time()
            probe()
        t0 = time.time()
        rows = read_rows(run / "actor.jsonl", replay)
        recent = rows[-a.recent:]
        end = random.randrange(a.window, len(recent) + 1)
        win = recent[end - a.window:end]
        inp = window_inputs(win)
        ids = inp["input_ids"][0]
        turns = turn_positions(ids.tolist(), tok, prefill="<|action_start|")
        if len(turns) < 3:
            continue
        d = dose_now()
        energy.enabled = d > 0
        energy.set_dose(cost_scale=d, noise_scale=d)
        energy.begin_step(seed=a.seed * 1_000_003 + step)
        warm = not pi.ready()  # fill the pair bank first, forward only
        # logits only where an action token is predicted (the head is 248k wide)
        keep = torch.tensor([p - 1 for t in turns for p in t["action"]], device=a.device)
        row_of = {int(p): i for i, p in enumerate(keep.tolist())}
        with torch.set_grad_enabled(not warm):
            out = model(**inp, output_hidden_states=True, logits_to_keep=keep)
        hs = out.hidden_states
        st = torch.tensor([t["state"] for t in turns], device=a.device)
        feats = {"s": hs[CENTRE][0, st],
                 "a": torch.stack([hs[CENTRE][0, t["action"]].mean(0) for t in turns]),
                 "rA": hs[REGION_A][0, st], "rB": hs[REGION_B][0, st]}
        with torch.set_grad_enabled(not warm):
            res = pi(feats)
        # log pi of each turn's sampled action tokens, over the restricted vocabulary
        logits = out.logits[0]
        del out
        lps, tok_lp, ents = [], [], []
        # one sequence over all action positions, so the output energy steps through them in order
        act_logits = output_energy(energy, logits[:, vocab].float(), cols_t) if cols else logits[:, vocab].float()
        for t in turns:
            pos = torch.tensor(t["action"], device=a.device)
            tgt = vocab_index[ids[pos]]
            ok = tgt >= 0
            rows_ = torch.tensor([row_of[int(p) - 1] for p in t["action"]], device=a.device)
            allowed = turn_mask(grammar, tgt.tolist()).to(a.device) if grammar is not None else None
            lp_all, plp = masked_log_probs(act_logits[rows_], allowed)
            if allowed is not None:
                lp_all = lp_all.clamp(min=-30.0)  # a target outside the mask (never sampled so) stays finite
            lp_tok = torch.where(ok, lp_all.gather(1, tgt.clamp(min=0)[:, None])[:, 0], torch.zeros_like(tgt, dtype=torch.float))
            lps.append(lp_tok[ok].mean())
            ents.append(-plp.sum(-1)[ok].mean())
            tok_lp.append(lp_tok)
        lps = torch.stack(lps)
        reward = res["reward"]
        r_mean = float(reward.mean()) if res["mi"] is not None else 0.0
        matched = 0
        if res["mi"] is not None:
            adv_mean = 0.99 * adv_mean + 0.01 * r_mean
            adv_var = 0.99 * adv_var + 0.01 * float(((reward - adv_mean) ** 2).mean())
            adv = ((reward - adv_mean) / (adv_var ** 0.5 + 1e-6)).clamp(-3, 3)
            # PPO: per-token ratio against the log-prob the actor sampled with, on the
            # tokens it actually sampled (the end tag after '<' is filled in, not sampled)
            terms = []
            for k, t in enumerate(turns[:-1]):
                sampled, old = win[k].get("sampled"), win[k].get("sampled_logp")
                if not sampled or ids[t["action"][:len(sampled)]].tolist() != sampled:
                    continue
                matched += 1
                new_lp = tok_lp[k][:len(sampled)]
                ratio = torch.exp(new_lp - torch.tensor(old, device=a.device))
                terms.append(-torch.min(ratio * adv[k], ratio.clamp(1 - a.ppo_eps, 1 + a.ppo_eps) * adv[k]).mean())
            loss_pg = torch.stack(terms).mean() if terms else torch.zeros((), device=a.device)
            loss_rep = -res["mi"]
        else:
            loss_pg = loss_rep = torch.zeros((), device=a.device)
        pr = participation_ratio(hs[CENTRE][0])
        lam = dual.lam
        floor = dual.floor or float(pr)
        loss_h = (lam * torch.relu(floor - pr) / floor) if dual.floor else torch.zeros((), device=a.device)
        ent = torch.stack(ents).mean()  # entropy of the action tokens, restricted vocabulary (nats)
        if dual_act.enabled and dual_act.floor:
            loss_h = loss_h + dual_act.lam * torch.relu(dual_act.floor - ent) / dual_act.floor
        loss = a.w_rep * loss_rep + a.w_pg * loss_pg + a.w_health * loss_h
        if "self_loss" in res:
            loss = loss + res["self_loss"]
        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], a.clip))
            opt.step()
        else:
            grad_norm = 0.0
        energy.end_step()
        step += 1
        dual.update(float(pr))
        dual_act.update(float(ent))
        if bored is not None and res["mi"] is not None:
            bored.update(r_mean)
        synced = None
        if step % a.merge_every == 0:
            changed = lora.merge()
            reset_adam_state(opt, list(lora.parameters()))
            path = sync.write(changed + list(direct), meta={"step": step})
            synced = path.name if path else None
            write_atomic(run / "dose.json", json.dumps({"dose": dose_now(), "step": step}))
        if time.time() - last_ckpt > a.checkpoint_every:
            last_ckpt = time.time()
            ck = run / "checkpoints"
            ck.mkdir(exist_ok=True)
            torch.save({n: t.detach().half().cpu() for n, t in sync.tensors.items()}, ck / f"step_{step:06d}.pt")
        row = {"t": round(time.time(), 3), "step": step, "warm": warm, "decision_end": win[-1]["n"],
               "newest_decision": rows[-1]["n"], "mi": None if res["mi"] is None else round(float(res["mi"]), 4),
               "reward_mean": round(r_mean, 4), "logp_mean": round(float(lps.mean()), 3),
               "loss_pg": round(float(loss_pg), 4), "ppo_turns": matched, "pr": round(float(pr), 3), "floor": dual.floor,
               "lam": round(lam, 4), "ent": round(float(ent), 4), "floor_act": dual_act.floor,
               "lam_act": round(dual_act.lam, 4), "dose": round(d, 4), "energy_changed": energy.changed_count() if d > 0 else 0,
               "grad_norm": round(grad_norm, 3), "tokens": int(ids.numel()), "seconds": round(time.time() - t0, 2), "synced": synced,
               "gpu_gib": round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2)}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if step <= 3 or step % 10 == 0:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
