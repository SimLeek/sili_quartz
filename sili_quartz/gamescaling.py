"""GameScaling-0.8B as an actor: system prompt, action parsing, policy, and
execution through robonet. Follows the GameScaling-0.8B-S3 model card
(Apache 2.0, THUSI-Lab): sections 2 (prompt), 4 (messages), 5 (actions).

One call = one 200 ms decision: the current frame plus up to 5 past
(frame, action) turns in, an action chunk out:

    <|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>

X, Y: mouse movement over the chunk in thousandths of the screen (X>0 right,
Y>0 down), Z: scroll notches; g1..g6: keys held in each 33 ms step.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

# -- system prompt (model card section 2; template text byte-for-byte from its example) --

_OPENING = (
    "You are a gaming expert, and you are currently playing {GAME_NAME}. You are proficient in keyboard "
    "and mouse operations. You understand game mechanics and combat pacing, can quickly extract key "
    "information from the current screen, and can think and make precise decisions at critical moments. "
    "Based on the current screen, plan the next 200ms of actions, consisting of 6 steps spaced 33ms apart. "
    "Each step lasts 33ms, until the next step begins. If the current situation continues the previous "
    "strategy, output the action directly. Think only when the situation changes significantly, the "
    "previous analysis is no longer valid, or a new objective appears."
)
_TAIL = """**Output Format**
**Action only (usual)**
<think></think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>
**Thinking + action (when necessary)**
<think>reasoning</think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>
**Explanation**
1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means move right, Y>0 means move down) and scroll amount Z (Z>0 means scroll up).
2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are separated by spaces, and groups are separated by semicolons.
- Each group can contain up to 4 keys.
- If a group has no keys, leave it empty but keep the `;`.
3. Only output a plain string that conforms to the above format — no line breaks and no quotation marks.
**Key Naming Rules**
- Number keys `0-9`: use lowercase English words, e.g., `zero` for `0`, `one` for `1`, ... `nine` for `9`.
- Function keys `F1-F12`: use capitalized English words, e.g., `One` represents `F1`, `Two` represents `F2`, and so on.
- Mouse buttons: use `LB`, `RB`, `MB` for the left, right, and middle mouse button.
- Letters: use the uppercase letter, e.g., `A`, `D`, `W`, `S`.
- Arrow keys: use `Up`, `Down`, `Left`, `Right`.
- Modifier/special keys: `Shift`, `Ctrl`, `Alt`, `Tab`, `Caps`, `Esc`, `Space`, `Enter`, `Back` (backspace), `Delete`, `Insert`, `Home`, `End`, `Pause`."""
MOUSE_AIM_LINE = "- Mouse X Y -> aim / rotate the camera (X>0 right, Y>0 down)"


def build_system_prompt(game: str, rules: str, key_bindings: list[tuple[str, str]], mouse_aim: bool = True) -> str:
    keys = [f"- {k} -> {v}" for k, v in key_bindings] + ([MOUSE_AIM_LINE] if mouse_aim else [])
    return "\n".join([
        _OPENING.replace("{GAME_NAME}", game),
        "**Game Rules**",
        rules,
        f"**Common Keys in {game}**",
        *keys,
        _TAIL,
    ])


# Track A is reward-free: the rules say what is on screen and how the game
# works, and give no goal or score.
MINECRAFT_RULES = """You are playing Minecraft (Java Edition 1.20.1, survival mode) in a single-player world.

## What you see on screen
- First-person view; the crosshair at screen centre marks the block or creature you would hit or use.
- The hotbar along the bottom shows nine item slots, with health (hearts) and hunger (drumsticks) above it.
- A minimap is in the top-left corner; a box at the top names the block under the crosshair.
- Tutorial hints may appear in the top-right corner.

## Rules
- Holding the attack button on a block breaks it after a moment and drops it as an item; walk over items to pick them up.
- Health falls from falls, drowning, lava, hunger and hostile creatures, which mostly appear at night and in the dark. At zero health you die and respawn.
- Items are crafted from other items in the inventory's crafting grid or at a crafting table."""

MINECRAFT_KEYS = [
    ("W", "Move forward"), ("A", "Strafe left"), ("S", "Move backward"), ("D", "Strafe right"),
    ("Space", "Jump; swim up"), ("Ctrl", "Sprint"), ("Shift", "Sneak"),
    ("LB", "Attack / break the block under the crosshair"),
    ("RB", "Use the held item / place a block / interact"),
    ("E", "Open or close the inventory"), ("Q", "Drop the held item"),
    ("one", "Select hotbar slot 1"), ("two", "Select hotbar slot 2"), ("three", "Select hotbar slot 3"),
    ("four", "Select hotbar slot 4"), ("five", "Select hotbar slot 5"), ("six", "Select hotbar slot 6"),
    ("seven", "Select hotbar slot 7"), ("eight", "Select hotbar slot 8"), ("nine", "Select hotbar slot 9"),
]

# -- actions (model card section 5) ---------------------------------------------------

_NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_FKEYS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"]
KEY_VOCAB = frozenset(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] + _NUMBERS + _FKEYS + ["LB", "RB", "MB"]
    + ["Up", "Down", "Left", "Right"]
    + ["Shift", "Ctrl", "Alt", "Tab", "Caps", "Esc", "Space", "Enter", "Back", "Delete", "Insert",
       "Home", "End", "Pause", "Equal", "Minus", "Period", "Slash", "Quote"]
)
assert len(KEY_VOCAB) == 74
MOUSE_BUTTONS = {"LB": "left", "RB": "right", "MB": "middle"}
# Training token -> robonet (pyautogui) key name. Alt, Pause and Caps have no robonet key.
TO_ROBONET = {
    **{chr(c): chr(c).lower() for c in range(ord("A"), ord("Z") + 1)},
    **{n: str(i) for i, n in enumerate(_NUMBERS)},
    **{f: f"f{i + 1}" for i, f in enumerate(_FKEYS)},
    "Up": "up", "Down": "down", "Left": "left", "Right": "right",
    "Shift": "shiftleft", "Ctrl": "ctrlleft", "Tab": "tab", "Esc": "esc", "Space": "space",
    "Enter": "enter", "Back": "backspace", "Delete": "delete", "Insert": "insert", "Home": "home",
    "End": "end", "Equal": "=", "Minus": "-", "Period": ".", "Slash": "/", "Quote": "'",
}
NOOP_TEXT = "<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>"
STEPS = 6
_BLOCK_RE = re.compile(r"<\|action_start\|>(.*?)<\|action_end\|>", re.DOTALL)
_BARE_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s*;.*$", re.DOTALL)


@dataclass
class Action:
    mouse: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: list[list[str]] = field(default_factory=lambda: [[] for _ in range(STEPS)])
    parsed: bool = False
    truncated: bool = False
    dropped: list[str] = field(default_factory=list)
    text: str = NOOP_TEXT  # what goes into the history (a no-op if unparsed)


def parse_action(response: str) -> Action:
    """Read the action after the last </think>; an unclosed <think> is a cut-off
    generation (no-op, truncated). Falls back to a bare `X Y Z ; ...`."""
    tail = response
    if "</think>" in response:
        tail = response.rsplit("</think>", 1)[1]
    elif "<think>" in response:
        return Action(truncated=True)
    m = _BLOCK_RE.search(tail)
    body = m.group(1) if m else (tail.split("<|action_end|>")[0] if _BARE_RE.match(tail.replace("<|action_start|>", "")) else None)
    if body is None:
        return Action()
    body = body.replace("<|action_start|>", "")
    parts = body.split(";")
    nums = parts[0].split()
    if len(nums) != 3:
        return Action()
    try:
        x, y, z = (float(v) for v in nums)
    except ValueError:
        return Action()
    x, y, z = max(-1000.0, min(1000.0, x)), max(-1000.0, min(1000.0, y)), max(-5.0, min(5.0, z))
    steps, dropped = [], []
    for group in (parts[1:] + [""] * STEPS)[:STEPS]:
        keys = []
        for k in group.split():
            (keys if k in KEY_VOCAB else dropped).append(k)
        steps.append(keys[:4])
    text = (f"<|action_start|>{x:g} {y:g} {z:g} ; " + " ; ".join(" ".join(s) for s in steps)
            + "<|action_end|>")
    return Action((x, y, z), steps, True, False, dropped, text)


# -- policy (model card section 4) --------------------------------------------------------

USER_TEXT = "current game screen"
PREFILL_DIRECT = "<think>\n\n</think>"
PREFILL_REASONING = "<think>"


class GameScalingPolicy:
    """Multi-turn: system; (user frame, assistant action) x history; user frame;
    assistant prefill. Past turns keep only the action. Needs a
    Qwen3_5ForConditionalGeneration and its processor (the checkpoint's own
    chat template)."""

    def __init__(self, model, processor, system_prompt: str, history_len: int = 5,
                 reasoning: bool = False, temperature: float = 1.0, max_new_tokens: int | None = None,
                 generator=None):
        self.model, self.processor, self.system = model, processor, system_prompt
        self.history: deque = deque(maxlen=history_len)
        self.prefill = PREFILL_REASONING if reasoning else PREFILL_DIRECT
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens or (2048 if reasoning else 96)
        tok = processor.tokenizer
        self.stop_ids = [tok.convert_tokens_to_ids("<|action_end|>")]
        if tok.eos_token_id is not None:
            self.stop_ids.append(tok.eos_token_id)
        self.last_seconds: float | None = None
        self.last_prompt_tokens: int | None = None

    def reset(self) -> None:
        self.history.clear()

    def messages(self, frame) -> list[dict]:
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]}]
        for past_frame, action_text in self.history:
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT},
                                                     {"type": "image", "image": past_frame}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": action_text}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT},
                                                 {"type": "image", "image": frame}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": self.prefill}]})
        return msgs

    def act(self, frame) -> tuple[Action, str]:
        """frame: a PIL image. Returns (action, raw completion)."""
        import torch

        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            self.messages(frame), tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=False, continue_final_message=True,
        ).to(self.model.device)
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                eos_token_id=self.stop_ids,
            )
        completion = self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        if "<|action_end|>" not in completion and "<|action_start|>" in completion:
            completion += "<|action_end|>"  # generate() stops on the token but may drop it on decode
        action = parse_action(self.prefill + completion)
        self.history.append((frame, action.text))
        self.last_seconds = time.perf_counter() - t0
        return action, completion


# -- execution through robonet ------------------------------------------------------------


class RobonetExecutor:
    """Plays an Action through a robonet RobonetAIClient: 6 steps of 33 ms,
    keys pressed/released on change, the chunk's mouse movement split evenly
    over the steps. Mouse moves are offsets from the screen centre (the game
    re-centres the pointer; Minecraft needs rawMouseInput:false). The frame
    must be the screen with no letterbox (brain ai_res 16:9)."""

    def __init__(self, client, step_s: float = 1.0 / 30.0):
        self.client, self.step_s = client, step_s
        self.held: set[str] = set()

    def _set_held(self, tokens: list[str]) -> None:
        want = {t for t in tokens if t in TO_ROBONET or t in MOUSE_BUTTONS}
        for t in self.held - want:
            self._send(t, False)
        for t in want - self.held:
            self._send(t, True)
        self.held = want

    def _send(self, token: str, pressed: bool) -> None:
        if token in MOUSE_BUTTONS:
            self.client.send_mouse_button(MOUSE_BUTTONS[token], pressed)
        else:
            self.client.send_key(TO_ROBONET[token], pressed)

    def execute(self, action: Action) -> None:
        x, y, z = action.mouse
        dx, dy = x / 1000.0 / STEPS, y / 1000.0 / STEPS
        if z:
            self.client.send_scroll(int(round(z)))
        for keys in action.steps:
            t0 = time.perf_counter()
            self._set_held(keys)
            if dx or dy:
                self.client.send_mouse_move(0.5 + dx, 0.5 + dy)
            time.sleep(max(0.0, self.step_s - (time.perf_counter() - t0)))

    def release_all(self) -> None:
        self._set_held([])
