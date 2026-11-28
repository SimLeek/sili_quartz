# Reasoning probes of the untouched model: base behaviors (2026-11-26)

What untouched GameScaling-0.8B says when allowed to think before acting on
real frames of its own play (`examples/reasoning_probe.py`, thinking mode,
one frame, no history). First 60 s test plus the baseline run
(`~/runs/a0_probes_esc_noxaero/` on the sandbox). Recorded now as the
reference for the trained arms; the predictions are the user's.

## Behaviors

| Behavior | Example | Prediction (user) |
|---|---|---|
| **Reads on-screen text literally, but not unfamiliar wording.** The menu was in pirate speak (it had changed `lang` to `enp`). | Link dialog with buttons "Open in Browser / Clove to Clipboard / Belay": "...open in the browser, clove it to the clipboard, or cancel, without the option to cancel." It copied "clove" and did not recognize "Belay" as cancel. | Training should fix "Belay": vocabulary from exposure. |
| **Invents goals.** | Same dialog: "Since the player already has the link and the goal involves opening it, I choose the browser option." Nothing on screen states a goal. | Not fixed by training; needs long-term memory (the planned large-memory block, "HDD-size" transformer block). |
| **Action does not match the plan.** | Plan: "moving onto Open in Browser and clicking LB". Action: `0 -1 0 ; LB ; LB ; LB`, a click almost in place, below the button. | Should improve with training: it should learn screen and mouse scale from its own play. |
| **Believes in progress it does not have.** | "The stone-axe recipe is prepared and the output is waiting..."; "the crafting grid now shows a crafted stick." It has no wood. | Expected for a small model in a new area: JEI shows recipes in crafting-window layouts, which look like real crafting. |
| **Reads the tutorial toast correctly.** | "a tutorial instruction now says to break down a tree by holding the Left Key." | Reference point. |
| **Decisions vary across samples.** | Two minutes earlier, on a near-identical frame: "I will avoid lingering in this menu and close it or press back to exit the screen." | Reference point. |
| **Copies empty thinking blocks.** With its 5-turn history in context (each past turn renders as an empty `<think></think>` before the action), it almost never thinks. | 0 thoughts in 15 samples with history vs about 1 in 6 without. | Probes use no history. Play is unaffected (the actor prefills an empty block anyway). |

## Avenues for later

- Re-probe the same frames (saved in `probe_frames/`) with each arm's final
  weights: does "Belay" become cancel, does the plan-action gap shrink?
- Invented goals as a test for the large-memory block once it exists.
- Thinking with history: strip the empty blocks from past turns, or keep a
  few real thoughts in the history, and see whether it thinks mid-stream.
