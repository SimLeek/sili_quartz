# Do the base model's thoughts change its actions? (2026-11-27)

`tools/thought_influence.py` on the untouched GameScaling-0.8B, P40, about 9
minutes, `~/runs/thought_check/base.jsonl` on the sandbox.

Setup:
- 12 single frames (no history) from each of three stretches: open world
  (A2g, minutes 0–10), open world (A2b, minutes 0–10), pause menu (A1-act,
  minutes 40–100).
- Prefill `<think>` and let the model think.
- When it does, 4 actions are sampled after its thought (T) and 4 after an
  empty thought block (D, how the actor plays). Every action is scored under
  both prefixes, giving a sequence-level KL estimate per action in each
  direction.

| set | states | thought | no thought | KL(D‖T) | KL(T‖D) | key overlap T–D | key overlap within D | thought tokens |
|---|---|---|---|---|---|---|---|---|
| open world A2g | 12 | 3 | 9 | 1.60 | 1.35 | 0.35 | 0.39 | 64 |
| open world A2b | 12 | 4 | 8 | 0.22 | 0.26 | 0.19 | 0.50 | 59 |
| pause menu A1-act | 12 | 4 | 8 | 0.53 | 1.03 | 0.26 | 0.21 | 65 |

With 4 samples per condition the KL estimates are noisy. Some per-state values
are negative (−0.42, −0.32), which only Monte Carlo noise can produce. The
per-state range is −0.4 to 4.1 nats.

Examples (all from open-world frames):

| thought (shortened) | keys after the thought | keys with no thought | KL(D‖T) |
|---|---|---|---|
| "...the pause menu has opened... use the mouse to return to the world" | Esc | none | −0.42 |
| "The unwanted inventory menu has moved into the browser... click the Cancel option with LB" | Alt, Shift | D, LB, Space | 1.11 |
| "...scroll down once more to select the axe, then aim at the trunk and hold LB" | eight, five (hotbar) | A, LB, Space, W | 4.10 |
| "The death screen is still waiting... click LB" | LB, Space | LB, Space | 0.70 |
| "...press Esc to return to control" | Esc, LB, Shift | Caps, Q | 0.21 |

## Reading

- **How often it thinks:** given the `<think>` prefill, it thought on 11 of 36
  frames and closed the block empty on the rest. When it thinks, it writes
  about 60–65 tokens.
- **Thoughts do move actions.** Several states shift by 1–4 nats per action,
  in the direction the thought says: "select the axe" leads to hotbar keys,
  "press Esc" to Esc, "click Respawn" to LB.
- **Thoughts are often wrong about the screen.** On open-world frames the
  model describes a pause menu, a death screen, an inventory "moved into the
  browser". That fits rationales written after the fact for ~6 % of training
  turns, describing screens the frame does not show.
- **So:** thoughts are a lever on actions, but the base model's thoughts are
  poorly grounded. Short thoughts trained together with the input tether and an
  RL policy would address both the grounding and the 60-token latency. The
  thought length is the knob to sweep (TODO 2d). The sample is small: 11
  thoughts, 4 actions per condition.
