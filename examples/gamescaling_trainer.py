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
          [+ lambda_act * relu(floor_act - H) / floor_act
           + lambda_act_hi * relu(H - ceil_act) / ceil_act with --health pr+act:
           H = entropy of the action tokens, held between 0.8x and 1/0.8x its
           early value; dose = max(lambda, lambda_act), the ceiling adds no dose]
          [+ the self-prediction loss for A1+self]
          [+ w_tether * the chunk tether (--tether chunks): per chunk of
           --tether-chunk layers, the average of the layers' outputs after
           acting on frame t matches the chunk's input at frame t+1]
          [+ w_tether * the output tether (--tether-out): at each emitted action
           token, the average of the last chunk's layer outputs matches the
           emitted token's embedding (the head is tied to the embeddings)]
          [+ lambda_in * the input-share band (--health pr+in, pr+act+in): how much of
           each chunk's output still points along its input; below its floor
           the energy dose is scaled down by share / floor]
          [--reward empower: the PPO reward is state empowerment instead of the
           pointwise MI: K probe actions from recent play are run from the
           window's last state, and the spread of the chunk tether's next-frame
           predictions across them, against its own error, is the channel
           capacity (sili_quartz/empowerment.py). It rewards the action that
           led to that state. A poor world model has a large error and so
           claims little control]
          [+ w_tether * the input tether (--tether head|col): the state after
           acting on frame t predicts frame t+1's mean image-token embedding,
           detached; head = a linear readout, col = each column of
           --tether-col-size neurons averages to one value of the input;
           a representation loss, never a reward];
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
    Boredom, ChunkTether, Dual, InputTether, OutputTether, energy_drive, PredictiveInfo, choice_columns, image_means, image_runs, input_share, make_energy, measure_io_means, measure_site_means,
    output_energy, participation_ratio, turn_positions,
)
from sili_quartz.continual import (
    ContinualLoRA, RowNormBand, WeightSync, direct_params, load_checkpoint, reset_adam_state, sync_tensors, target_linears,
)
from sili_quartz.empowerment import state_empowerment
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
    ap.add_argument("--energy-io-side", choices=("both", "in"), default="both",
                    help="with --energy-io: both = input and output neurons; in = input neurons only, no energy "
                         "on the action choice (actor and trainer must match)")
    ap.add_argument("--energy-io", action="store_true",
                    help="energy also on the input neurons (token embeddings, vision merger) and the output "
                         "neurons (action-choice probabilities); the actor must run with --energy-io too")
    ap.add_argument("--health", choices=("pr", "pr+in", "pr+act", "pr+act+in"), default="pr",
                    help="pr+act: also hold the action entropy between 0.8x and 1/0.8x its early value "
                         "(second dual, two-sided; a direct loss on output randomness, superseded); "
                         "+in: the input-share band")
    ap.add_argument("--grammar", action="store_true",
                    help="the actor sampled under the action-format mask: apply the same mask to log pi and "
                         "the entropy")
    ap.add_argument("--tether", choices=("none", "head", "col", "chunks"), default="none",
                    help="input tether: the centre state after acting predicts the next frame's input "
                         "(head: linear readout; col: neural columns, each column's average matches one value)")
    ap.add_argument("--w-tether", type=float, default=1.0)
    ap.add_argument("--tether-col-size", type=int, default=8, help="neurons per column for --tether col")
    ap.add_argument("--tether-chunk", type=int, default=4,
                    help="layers per chunk for --tether chunks and the input-share guard")
    ap.add_argument("--tether-chunks", type=int, default=None,
                    help="tether only the first N chunks, counting from the input side (default all); the "
                         "chunks nearest the output are left to the policy")
    ap.add_argument("--tether-out", action="store_true",
                    help="output tether: the last chunk's layer average matches each emitted action token")
    ap.add_argument("--w-tether-out", type=float, default=None,
                    help="weight of the output tether (default: --w-tether); the policy term should dominate "
                         "the output side")
    ap.add_argument("--renorm", action="store_true",
                    help="amortized magnitude health: after each merge, rescale rows of 1/8 of the Linears "
                         "back into [0.8, 1.25] x their starting norms")
    ap.add_argument("--reward", choices=("mi", "empower", "energy"), default="mi",
                    help="PPO reward: the pointwise MI of the arm; state empowerment (needs --tether chunks); "
                         "or energy curiosity: the drive reduction D(frame k) - D(frame k+1), D = share of the "
                         "input-side neurons energy forced or shut off on the frame (needs a fixed dose > 0)")
    ap.add_argument("--reward-layers", type=int, default=12,
                    help="--reward energy: the first N layers' MLP and attention sites are the input side")
    ap.add_argument("--probes", type=int, default=4, help="probe actions per empowerment estimate")
    ap.add_argument("--init", type=Path, default=None,
                    help="start from a trainer checkpoint, e.g. a previous run's last; the actor needs the same "
                         "--init. The health anchors (row-norm band, floors) are measured from these weights")
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
    if a.init is not None:
        print(json.dumps({"init": str(a.init), "tensors": load_checkpoint(model, a.init)}), flush=True)
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
    n_layers = len(model.model.language_model.layers)
    if a.tether == "chunks":
        tether = ChunkTether(n_layers, chunk=a.tether_chunk, n_tethered=a.tether_chunks).to(a.device)
    elif a.tether != "none":
        tether = InputTether(a.tether, col_size=a.tether_col_size, seed=a.seed).to(a.device)
    else:
        tether = None
    band = RowNormBand(linears) if a.renorm else None
    tether_out = OutputTether(n_layers, chunk=a.tether_chunk) if a.tether_out else None
    image_token = model.config.image_token_id
    opt = torch.optim.AdamW([
        {"params": list(lora.parameters()), "lr": a.lr},
        {"params": list(direct.values()), "lr": a.lr_direct, "weight_decay": 0.0},
        {"params": list(pi.parameters()) + (list(tether.parameters()) if tether else []), "lr": 1e-3},
    ])
    sync = WeightSync(sync_tensors(linears, direct), run / "sync")
    dual = Dual(enabled=a.arm != "A1-lam")
    bored = Boredom() if a.arm == "A1+bored" else None
    dual_act = Dual(enabled=a.health in ("pr+act", "pr+act+in") and a.arm != "A1-lam", two_sided=True)
    dual_in = Dual(enabled=a.health in ("pr+in", "pr+act+in") and a.arm != "A1-lam", two_sided=True)
    share_now = [1.0]

    def dose_now() -> float:
        if a.dose == "none" or a.arm == "A1-lam":
            return 0.0
        if a.dose.startswith("fixed:"):
            return float(a.dose.split(":")[1])
        d = max(dual.lam, dual_act.lam) + (bored.extra if bored else 0.0)
        if dual_in.enabled and dual_in.floor:
            d *= min(1.0, share_now[0] / dual_in.floor)  # less dreaming while the input share is low
        return d

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
    cols = choice_columns(tok, vocab.tolist()) if a.energy_io and a.energy_io_side == "both" else []
    if a.reward == "energy":
        assert a.dose.startswith("fixed:") and float(a.dose.split(":")[1]) > 0, "--reward energy needs --dose fixed:x"
    cols_t = torch.tensor(cols, device=a.device)
    energy = make_energy(model, mlp_means, attn_means, io_means=io_means if a.energy_io else None,
                         n_choice=len(cols))
    if a.reward == "energy":  # sites: MLP outputs per layer, then attention outputs per layer
        nl = len(mlp_means)
        energy.trace_sites = set(range(a.reward_layers)) | {nl + j for j in range(a.reward_layers)}
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
        emp, emp_st = None, None
        if a.reward == "empower" and isinstance(tether, ChunkTether) and len(turns) == len(win):
            emp_st = tether.chunk0_stats()
            pool = list(dict.fromkeys(r["raw"] for r in recent if r.get("parsed")
                                      and (r.get("raw") or "").rstrip().endswith(ACTION_END)))
            # no gate on emp_st["ready"]: a poor world model has a large sigma2 and so claims little control
            if emp_st is not None and len(pool) >= a.probes:
                hist = [(Image.open(replay / f"{r['n']:06d}.jpg").convert("RGB"), assistant_content(r)) for r in win[:-1]]
                frame = Image.open(replay / f"{win[-1]['n']:06d}.jpg").convert("RGB")
                energy.enabled = False  # the probe measures the world model's channel, not energy's noise
                model.eval()  # gradient checkpointing turns the cache off in train mode
                emp, _ = state_empowerment(model, proc, system, hist, frame, random.sample(pool, a.probes),
                                           a.tether_chunk, emp_st)
                model.train()
        d = dose_now()
        energy.enabled = d > 0
        energy.set_dose(cost_scale=d, noise_scale=d)
        # start from the actor's energy state at (or just before) the window, not from rest:
        # the actor never resets its energy during a session
        snap = None
        if d > 0:
            snaps = sorted(replay.glob("energy_*.pt"))
            older = [q for q in snaps if int(q.stem.split("_")[1]) <= win[0]["n"]]
            snap = older[-1] if older else None
        if snap is not None:
            try:
                energy.load_states(torch.load(snap, map_location="cpu", weights_only=True), a.device)
            except (OSError, RuntimeError, EOFError):
                snap = None  # a file still being written
        energy.begin_step(seed=a.seed * 1_000_003 + step, at_rest=snap is None)
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
            tres = None
            if isinstance(tether, ChunkTether):
                tres = tether(hs, [t["action"][-1] for t in turns],
                              image_runs(ids.tolist(), image_token, [t["state"] for t in turns]))
            elif tether is not None:
                after = hs[CENTRE][0, torch.tensor([t["action"][-1] for t in turns], device=a.device)]
                tres = tether(after, image_means(ids.tolist(), hs[0][0], image_token, [t["state"] for t in turns]))
            ores = None
            if tether_out is not None:
                qs = [q for t in turns for q in t["action"]]
                emitted = model.get_input_embeddings().weight[ids[qs]]
                ores = tether_out(hs, qs, emitted)
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
        # rewards by turn index: the arm's pointwise MI for every turn but the last, or the
        # empowerment of the window's last state for the action that led to it
        drives = None
        if a.reward == "energy" and energy.trace:
            drives = energy_drive(energy.trace, image_runs(ids.tolist(), image_token, [t["state"] for t in turns]))
            energy.trace = {}
        if a.reward == "mi":
            rewards = {k: float(v) for k, v in enumerate(res["reward"])} if res["mi"] is not None else {}
        elif a.reward == "energy":
            rewards = {k: drives[k] - drives[k + 1] for k in range(len(turns) - 1)} if drives and not warm else {}
        else:
            rewards = {len(turns) - 2: emp} if emp is not None and not warm else {}
        r_mean = sum(rewards.values()) / len(rewards) if rewards else 0.0
        matched = 0
        loss_rep = -res["mi"] if res["mi"] is not None else torch.zeros((), device=a.device)
        loss_pg = torch.zeros((), device=a.device)
        if rewards:
            adv_mean = 0.99 * adv_mean + 0.01 * r_mean
            adv_var = 0.99 * adv_var + 0.01 * sum((v - adv_mean) ** 2 for v in rewards.values()) / len(rewards)
            adv = {k: max(-3.0, min(3.0, (v - adv_mean) / (adv_var ** 0.5 + 1e-6))) for k, v in rewards.items()}
            # PPO: per-token ratio against the log-prob the actor sampled with, on the
            # tokens it actually sampled (the end tag after '<' is filled in, not sampled)
            terms = []
            for k, t in enumerate(turns[:-1]):
                if k not in adv:
                    continue
                sampled, old = win[k].get("sampled"), win[k].get("sampled_logp")
                if not sampled or ids[t["action"][:len(sampled)]].tolist() != sampled:
                    continue
                matched += 1
                new_lp = tok_lp[k][:len(sampled)]
                ratio = torch.exp(new_lp - torch.tensor(old, device=a.device))
                terms.append(-torch.min(ratio * adv[k], ratio.clamp(1 - a.ppo_eps, 1 + a.ppo_eps) * adv[k]).mean())
            loss_pg = torch.stack(terms).mean() if terms else torch.zeros((), device=a.device)
        pr = participation_ratio(hs[CENTRE][0])
        lam = dual.lam
        floor = dual.floor or float(pr)
        loss_h = (lam * torch.relu(floor - pr) / floor) if dual.floor else torch.zeros((), device=a.device)
        ent = torch.stack(ents).mean()  # entropy of the action tokens, restricted vocabulary (nats)
        if dual_act.enabled and dual_act.floor:
            loss_h = loss_h + dual_act.penalty(ent)
        share = None
        if dual_in.enabled:
            shares = input_share(hs, a.tether_chunk)
            share = shares.mean()
            if dual_in.floor:
                loss_h = loss_h + dual_in.penalty(share)
        loss = a.w_rep * loss_rep + a.w_pg * loss_pg + a.w_health * loss_h
        if "self_loss" in res:
            loss = loss + res["self_loss"]
        if tres is not None and not warm:
            loss = loss + a.w_tether * tres["tether_loss"]
        if ores is not None and not warm:
            loss = loss + (a.w_tether if a.w_tether_out is None else a.w_tether_out) * ores["tether_out_loss"]
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
        if share is not None:
            share_now[0] = float(share)
            dual_in.update(float(share))
        if bored is not None and res["mi"] is not None:
            bored.update(r_mean)
        synced = None
        if step % a.merge_every == 0:
            changed = lora.merge()
            reset_adam_state(opt, list(lora.parameters()))
            renormed = 0
            if band is not None:
                moved, renormed = band.step()
                changed = sorted(set(changed) | set(moved))
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
               "energy_snapshot": None if snap is None else int(snap.stem.split("_")[1]),
               "lam_act": round(dual_act.lam, 4), "ceil_act": dual_act.ceiling, "lam_act_hi": round(dual_act.lam_hi, 4), "dose": round(d, 4), "energy_changed": energy.changed_count() if d > 0 else 0,
               "grad_norm": round(grad_norm, 3), "tokens": int(ids.numel()), "seconds": round(time.time() - t0, 2), "synced": synced,
               "gpu_gib": round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2)}
        if tres is not None:
            row["tether_mse"] = round(tres["tether_mse"], 4)
            row["tether_copy_mse"] = round(tres["tether_copy_mse"], 4)
            if "tether_chunk_mse" in tres:
                row["tether_chunk_mse"] = tres["tether_chunk_mse"]
                row["tether_chunk_copy_mse"] = tres["tether_chunk_copy_mse"]
        if ores is not None:
            row["tether_out_mse"] = round(ores["tether_out_mse"], 4)
        if drives is not None:
            row["drive"] = [round(x, 4) for x in drives]
        if emp_st is not None:
            row.update({"emp": None if emp is None else round(emp, 4), "emp_ready": emp_st["ready"],
                        "chunk0_mse": round(emp_st["mse"], 4), "chunk0_copy_mse": round(emp_st["copy_mse"], 4)})
        if share is not None:
            row.update({"share": round(float(share), 4), "share_chunks": [round(float(x), 4) for x in shares],
                        "floor_in": dual_in.floor, "ceil_in": dual_in.ceiling, "lam_in": round(dual_in.lam, 4),
                        "lam_in_hi": round(dual_in.lam_hi, 4)})
        if band is not None and step % a.merge_every == 0:
            row["renorm_rows"] = renormed
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if step <= 3 or step % 10 == 0:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
