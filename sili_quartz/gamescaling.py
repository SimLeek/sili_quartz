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
- Items are crafted from other items in the inventory's crafting grid or at a crafting table.
- If you see a "You Died!" screen, move the mouse onto the Respawn button and click it with LB. Then you can play again.
- On a menu screen the mouse moves an arrow. Move the arrow onto a button and click LB to press it."""

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


def action_vocab(tokenizer, samples: int = 4000, seed: int = 0) -> list[int]:
    """Token ids that valid action text uses, from tokenizing many random
    valid actions (numbers, key groups, single and double spaces) plus the end
    tag and end of turn. Decoding restricted to these is sampling from the
    model's distribution conditioned on staying in the action format."""
    import random

    rng = random.Random(seed)
    keys = sorted(KEY_VOCAB)
    ids: set[int] = set()
    for i in range(samples):
        x, y = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
        z = rng.choice([0, 0, 0, rng.randint(-5, 5)])
        groups = [" ".join(rng.sample(keys, rng.randint(0, 4))) for _ in range(STEPS)]
        sep = rng.choice([" ; ", " ;  ", ";", " ;"])
        text = f"{x} {y} {z}{sep}" + sep.join(groups) + ACTION_END
        ids.update(tokenizer.encode(text, add_special_tokens=False))
    for t in (ACTION_END, "<|im_end|>", ">", ">-"):  # ">" / ">-" finish the prefilled tag
        ids.update(tokenizer.encode(t, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


_NUM_PARTIAL = re.compile(r"[-+]?(?:\d{1,4}(?:\.\d{0,2})?)?")  # no dead ends: a digit before any "."
_NUM_FULL = re.compile(r"[-+]?\d{1,4}(?:\.\d{1,2})?")
_KEY_PREFIXES = frozenset(k[:i] for k in KEY_VOCAB for i in range(1, len(k) + 1))


def action_prefix_state(s: str) -> str | None:
    """s: decoded text after the '<|action_start|' prefill. None if no valid
    action starts with s, 'done' if s is a complete action (it ends in the
    start of the end tag), else 'open'. Valid: '>' X Y Z ';' then up to STEPS
    ';'-separated groups of at most 4 known keys, closed by '<'."""
    if not s:
        return "open"
    if s[0] != ">" or "    " in s:  # runs of 4+ spaces only spend the token budget
        return None
    r = s[1:]
    head, semi, rest = r.partition(";")
    if any(c not in "0123456789-+. " for c in head):
        return None
    nums = head.split()
    closed = nums if head.endswith(" ") else nums[:-1]  # numbers already ended by a space
    if len(nums) > 3 or not all(_NUM_PARTIAL.fullmatch(n) for n in nums) or \
            not all(_NUM_FULL.fullmatch(n) for n in closed):
        return None
    if not semi:
        return "open"
    if len(nums) != 3 or not all(_NUM_FULL.fullmatch(n) for n in nums):
        return None
    body, lt, tag = rest.partition("<")
    if body.count(";") > STEPS - 1:
        return None
    for group in body.split(";"):
        if any(not (c.isalnum() or c == " ") for c in group):
            return None
        words = group.split()
        if len(words) > 4:
            return None
        closed = words if group.endswith(" ") or not words else words[:-1]
        if any(w not in KEY_VOCAB for w in closed) or (words and words[-1] not in _KEY_PREFIXES):
            return None
    # a word still being typed when the group closes must be a whole key
    for group in body.split(";")[:-1] + ([body.split(";")[-1]] if lt else []):
        if group.split() and group.split()[-1] not in KEY_VOCAB:
            return None
    if not lt:
        return "open"
    return "done" if ACTION_END.startswith("<" + tag) else None


class ActionGrammar:
    """Masks the action vocabulary so every sampled completion parses: a
    token is allowed if the text stays a prefix of a valid action. Sampling
    under the mask is the model's distribution conditioned on the format;
    the trainer applies the same mask, so PPO's ratio and the entropy are
    over the same distribution the actor sampled from."""

    def __init__(self, tokenizer, vocab: list[int], max_new_tokens: int = 64):
        self.strings = [tokenizer.decode([t], skip_special_tokens=False) for t in vocab]
        self.end_ok = [s.lstrip(" ").startswith("<") and ACTION_END.startswith(s.lstrip(" ")) for s in self.strings]
        self.max_new_tokens = max_new_tokens
        self._cache: dict[tuple[str, bool], list[bool]] = {}

    def allowed(self, text: str, made: int) -> list[bool]:
        """Allowed vocab positions after `text` (decoded so far, `made` tokens)."""
        last = made >= self.max_new_tokens - 1  # one token left: it must close the action
        key = (text, last)
        if key not in self._cache:
            ok = []
            for s, end in zip(self.strings, self.end_ok):
                st = action_prefix_state(text + s)
                ok.append(st == "done" if last else (st is not None and (st != "done" or end)))
            if not any(ok):  # cannot happen for a valid prefix; never mask everything
                ok = [True] * len(ok)
            if len(self._cache) > 50_000:
                self._cache.clear()
            self._cache[key] = ok
        return self._cache[key]

    def mask(self, text: str, made: int, device=None):
        import torch

        return torch.tensor(self.allowed(text, made), dtype=torch.bool, device=device)


MASKED_LOGIT = -1e4  # finite: a -inf fill gives exp(lp) * lp = 0 * -inf = NaN in backward


def masked_log_probs(logits, allowed=None):
    """(log p, p log p) over the last dim, restricted to `allowed` (bool, same
    shape) when given. Masked entries get log p = MASKED_LOGIT-ish and
    p log p = 0, both with finite gradients; the distribution over allowed
    tokens matches a -inf mask exactly in fp32."""
    import torch

    if allowed is not None:
        logits = logits.masked_fill(~allowed, MASKED_LOGIT)
    lp = torch.log_softmax(logits, -1)
    plp = lp.exp() * lp
    if allowed is not None:
        plp = torch.where(allowed, plp, torch.zeros_like(plp))
    return lp, plp


# -- policy (model card section 4) --------------------------------------------------------

USER_TEXT = "current game screen"
# The chat template renders a past turn without reasoning as
# "<think>\n\n</think>\n\n" + action, so prefilling the fixed action tag
# matches training and saves generating it -- but only up to its last
# character: the tokenizer merges the tag's ">" with a negative X ("">-"",
# one token), so a prefill ending in ">" makes moving left impossible.
PREFILL_DIRECT = "<think>\n\n</think>\n\n<|action_start|"
PREFILL_REASONING = "<think>"
ACTION_END = "<|action_end|>"


class GameScalingPolicy:
    """Multi-turn: system; (user frame, assistant action) x history; user frame;
    assistant prefill. Past turns keep only the action. Needs a
    Qwen3_5ForConditionalGeneration and its processor (the checkpoint's own
    chat template).

    session_turns > 0 keeps the KV cache across decisions: each turn only the
    new user turn (one frame) is run through the model, at the positions it
    has in the full conversation. The conversation then grows past
    history_len; after session_turns decisions (or whenever the cache no
    longer matches the rendered conversation) it starts over from the last
    history_len turns. session_turns = 0: every call re-reads the last
    history_len turns (the model card's setup).
    """

    def __init__(self, model, processor, system_prompt: str, history_len: int = 5,
                 reasoning: bool = False, temperature: float = 1.0, max_new_tokens: int | None = None,
                 session_turns: int = 0, generator=None, fast_decode: bool = False, max_len: int = 16384):
        self.model, self.processor, self.system = model, processor, system_prompt
        self.history_len = history_len
        self.turns: list = []  # (frame, assistant content) for the current session
        self.prefill = PREFILL_REASONING if reasoning else PREFILL_DIRECT
        self.temperature = temperature
        self.logit_post = None  # callable(logits) -> logits before sampling, e.g. output energy
        self.grammar = None  # ActionGrammar: sample only completions that parse
        self.max_new_tokens = max_new_tokens or (2048 if reasoning else 64)
        self.session_turns = session_turns
        self.generator = generator
        self.tok = processor.tokenizer
        self.decoder = None
        if fast_decode:
            from sili_quartz.fast_decode import GraphDecoder

            self.vocab = action_vocab(self.tok)
            self.decoder = GraphDecoder(model, max_len=max_len, vocab_ids=self.vocab)
        self._reset_session()
        self.last_seconds: float | None = None
        self.last_prompt_tokens: int | None = None
        self.last_new_tokens: int | None = None
        self.last_reused: bool = False
        self.sampled_logp: list[float] = []  # per sampled token, the last decision
        self.sampled_tokens: list[int] = []

    @property
    def history(self):
        return self.turns[-self.history_len:] if self.history_len else []

    def reset(self) -> None:
        self.turns.clear()
        self._reset_session()

    def _reset_session(self) -> None:
        self.cache = None
        self.cache_text = ""  # the conversation text the cache holds
        self.cache_ids: list[int] = []
        self.grids: list = []  # image_grid_thw rows of the images in the cache
        self.mm_types: list[int] = []
        self.session_count = 0

    def messages(self, frame, turns=None) -> list[dict]:
        turns = self.history if turns is None else turns
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]}]
        for past_frame, content in turns:
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT},
                                                     {"type": "image", "image": past_frame}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT},
                                                 {"type": "image", "image": frame}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": self.prefill}]})
        return msgs

    def _render(self, msgs) -> str:
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                                  continue_final_message=True)

    def _positions(self, ids: list[int]):
        """M-RoPE positions (3, 1, L) for the whole conversation, and its delta."""
        import torch

        dev = self.model.device
        return self.model.model.get_rope_index(
            torch.tensor([ids], device=dev),
            mm_token_type_ids=torch.tensor([self.mm_types], device=dev),
            image_grid_thw=torch.cat(self.grids).to(dev),
        )

    def _prefill(self, frame) -> tuple:
        """Feed the model whatever the cache lacks. Returns the last logits."""
        import torch

        session_turns = self.turns if self.session_turns else self.history
        full_text = self._render(self.messages(frame, session_turns))
        reuse = (self.session_turns and self.cache is not None and self.session_count < self.session_turns
                 and full_text.startswith(self.cache_text))
        if not reuse:
            self._reset_session()
            session_turns = self.history
            full_text = self._render(self.messages(frame, session_turns))
            frames = [f for f, _ in session_turns] + [frame]
            suffix_text = full_text
        else:
            frames = [frame]
            suffix_text = full_text[len(self.cache_text):]
        inputs = self.processor(text=[suffix_text], images=frames, return_tensors="pt").to(self.model.device)
        new_ids = inputs["input_ids"][0].tolist()
        start = len(self.cache_ids)
        self.cache_ids += new_ids
        self.mm_types += inputs["mm_token_type_ids"][0].tolist()
        self.grids.append(inputs["image_grid_thw"].cpu())
        positions, self.rope_delta = self._positions(self.cache_ids)
        out = self.model(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=inputs["mm_token_type_ids"],
            position_ids=positions[:, :, start:], past_key_values=self.cache, use_cache=True,
        )
        self.cache = out.past_key_values
        self.cache_text = full_text
        self.last_reused, self.last_new_tokens = bool(reuse), len(new_ids)
        self.last_prompt_tokens = len(self.cache_ids)
        return out.logits[:, -1]

    def _sample(self, logits, gtext: str = "", made: int = 0) -> int:
        """Sample an index into logits; records its log-probability (at the
        sampling temperature) in self.sampled_logp for PPO's ratio. With a
        grammar, logits outside the action format are masked first (gtext:
        the vocab strings sampled so far, made: how many)."""
        import torch

        if self.logit_post is not None:
            logits = self.logit_post(logits)
        if self.grammar is not None:
            logits = logits.masked_fill(~self.grammar.mask(gtext, made, logits.device), float("-inf"))
        if self.temperature > 0:
            logp = (logits.float() / self.temperature).log_softmax(-1)
            idx = int(torch.multinomial(logp.exp(), 1, generator=self.generator))
        else:
            logp = logits.float().log_softmax(-1)
            idx = int(logits.argmax(-1))
        self.sampled_logp.append(float(logp[idx]))
        return idx

    def _decode_fast(self, logits) -> str:
        """As _decode, on the CUDA-graph decoder, sampling inside the action
        vocabulary, and stopping at the '<' that starts the end tag (the rest
        of the tag reaches the cache with the next prefill)."""
        dec = self.decoder
        dec.load(self.cache, len(self.cache_ids), int(self.rope_delta.flatten()[0]))
        vocab = self.vocab
        logits = logits[0, dec.vocab_ids]
        made: list[int] = []
        self.sampled_logp, self.sampled_tokens = [], made
        text = ""
        gtext = ""
        for _ in range(self.max_new_tokens):
            i = self._sample(logits, gtext, len(made))
            token = vocab[i]
            made.append(token)
            text = self.tok.decode(made, skip_special_tokens=False)
            if self.grammar is not None:
                gtext += self.grammar.strings[i]
            if token == self.tok.eos_token_id or ACTION_END in text or (
                    text.count(";") >= STEPS and "<" in text.rsplit(";", 1)[-1]) or (
                    self.grammar is not None and action_prefix_state(gtext) == "done"):
                break
            self.cache_ids.append(token)
            self.mm_types.append(0)
            logits = dec.step(token)[0]
        dec.store(self.cache)
        # the final token is not in the cache: keep the text the cache holds
        held = self.tok.decode(made[:-1], skip_special_tokens=False)
        if ACTION_END not in text and ";" in text:
            text = text[: text.rfind("<")] + ACTION_END if "<" in text.rsplit(";", 1)[-1] else text
        self.cache_text += held
        return text

    def _decode(self, logits) -> str:
        """Sample until the action's end tag (or the token budget)."""
        import torch

        if self.decoder is not None:
            return self._decode_fast(logits)
        dev = self.model.device
        made: list[int] = []
        text = ""
        for _ in range(self.max_new_tokens):
            if self.temperature > 0:
                probs = (logits.float() / self.temperature).softmax(-1)
                token = int(torch.multinomial(probs, 1, generator=self.generator))
            else:
                token = int(logits.argmax(-1))
            made.append(token)
            self.cache_ids.append(token)
            self.mm_types.append(0)
            text = self.tok.decode(made, skip_special_tokens=False)
            if ACTION_END in text or token == self.tok.eos_token_id:
                break
            pos = torch.full((3, 1, 1), len(self.cache_ids) - 1, device=dev) + self.rope_delta.to(dev)
            out = self.model(input_ids=torch.tensor([[token]], device=dev), position_ids=pos,
                             past_key_values=self.cache, use_cache=True)
            self.cache, logits = out.past_key_values, out.logits[:, -1]
        self.cache_text += text
        return text

    def act(self, frame) -> tuple[Action, str]:
        """frame: a PIL image. Returns (action, raw completion)."""
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._prefill(frame)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            t1 = time.perf_counter()
            n_before = len(self.cache_ids)
            completion = self._decode(logits)
        self.last_prefill_s, self.last_decode_s = t1 - t0, time.perf_counter() - t1
        self.last_generated = len(self.cache_ids) - n_before + 1
        action = parse_action(self.prefill + completion)
        # Past turns render as "<think>\n\n</think>\n\n" + content: keep what was
        # generated (so the cache still matches) when it parsed, else a no-op.
        if action.parsed and completion.rstrip().endswith(ACTION_END):
            content = self.prefill.split("</think>\n\n", 1)[-1] + completion
        else:
            content = NOOP_TEXT
        self.turns.append((frame, content))
        self.session_count += 1
        if not self.session_turns:
            self.turns = self.turns[-self.history_len:]
        self.last_seconds = time.perf_counter() - t0
        return action, completion


class StreamingPolicy(GameScalingPolicy):
    """Constant cost per decision, for unbounded play: only the new frame turn
    is run through the model, and decoding is the CUDA-graph decoder.

    Qwen3.5 is mostly recurrent: the 18 linear-attention layers carry the
    whole past in fixed-size state. The 6 full-attention layers keep a
    sliding window of KV: the system prompt (an attention sink) plus the
    last kv_window tokens (about history_len turns); older KV is dropped
    after each decision. Positions keep counting up, so nothing is
    renumbered: each turn's M-RoPE positions are computed on the turn alone
    and shifted (M-RoPE is shift-equivariant). Each turn's text is the
    template's per-turn piece, so no conversation is ever re-rendered.
    """

    MARKER = "\x00CONTENT\x00"

    def __init__(self, model, processor, system_prompt: str, kv_window: int | None = None,
                 history_len: int = 5, temperature: float = 1.0, generator=None, max_len: int = 8192,
                 max_new_tokens: int | None = None, keep_transcript: bool = False):
        super().__init__(model, processor, system_prompt, history_len=history_len, temperature=temperature,
                         max_new_tokens=max_new_tokens, session_turns=0, generator=generator,
                         fast_decode=True, max_len=max_len)
        from PIL import Image

        dummy = Image.new("RGB", (64, 64))
        self.first_text = self._render(self.messages(dummy, []))
        two = self._render(self.messages(dummy, [(dummy, self.MARKER)]))
        self.turn_text = two.split(self.MARKER, 1)[1]  # after a past action: end of turn ... new prefill
        system_part = self.first_text.split("<|im_start|>user", 1)[0]
        self.sink = len(self.tok(system_part, add_special_tokens=False).input_ids)
        self.kv_window = kv_window
        self.per_turn: int | None = None  # tokens one turn adds (measured on the first)
        self.keep_transcript = keep_transcript  # for checks: every text fed, and the frames
        self.reset()

    def reset(self) -> None:
        self.turns = []
        self.cache = None
        self.kv_len = 0
        self.next_pos = 0
        self.owed = ""  # text of the last action that is not in the cache yet
        self.transcript: list[str] = []
        self.frames: list = []

    def _attention_layers(self):
        return [i for i, t in enumerate(self.decoder.types) if t != "linear_attention"]

    def _trim(self) -> None:
        window = self.kv_window or (self.history_len * (self.per_turn or 300))
        if self.kv_len <= self.sink + window:
            return
        for i in self._attention_layers():
            layer = self.cache.layers[i]
            layer.keys = torch_cat_keep(layer.keys, self.sink, window)
            layer.values = torch_cat_keep(layer.values, self.sink, window)
        self.kv_len = self.sink + window

    def act(self, frame) -> tuple[Action, str]:
        import torch

        t0 = time.perf_counter()
        dev = self.model.device
        first = self.cache is None
        text = self.first_text if first else self.owed + self.turn_text
        with torch.no_grad():
            inputs = self.processor(text=[text], images=[frame], return_tensors="pt").to(dev)
            n = int(inputs["input_ids"].shape[1])
            rel, _ = self.model.model.get_rope_index(
                inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                image_grid_thw=inputs["image_grid_thw"])
            positions = rel + self.next_pos
            out = self.model(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=inputs["mm_token_type_ids"],
                position_ids=positions, past_key_values=self.cache, use_cache=True)
            self.cache = out.past_key_values
            self.kv_len += n
            self.next_pos = int(positions.max()) + 1
            self.last_positions = positions
            if self.keep_transcript:
                self.transcript.append(text)
                self.frames.append(frame)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            completion, fed = self._decode_stream(out.logits[0, -1])
        action = parse_action(self.prefill + completion)
        self.turns.append((None, action.text))
        self.turns = self.turns[-self.history_len:]
        if first:
            self.per_turn = None
        elif self.per_turn is None:
            self.per_turn = n + fed
        self._trim()
        self.last_prefill_s, self.last_decode_s = t1 - t0, time.perf_counter() - t1
        self.last_seconds = time.perf_counter() - t0
        self.last_new_tokens, self.last_prompt_tokens, self.last_generated = n, self.kv_len, fed + 1
        self.last_reused = not first
        return action, completion

    def _decode_stream(self, logits) -> tuple[str, int]:
        """Sample in the action vocabulary until the '<' of the end tag. The
        last token is not fed; what it and the end tag owe is fed with the
        next turn."""
        dec = self.decoder
        dec.load(self.cache, self.kv_len, self.next_pos - self.kv_len)
        logits = logits[dec.vocab_ids]
        made: list[int] = []
        self.sampled_logp, self.sampled_tokens = [], made
        text = ""
        gtext = ""
        for _ in range(self.max_new_tokens):
            i = self._sample(logits, gtext, len(made))
            token = self.vocab[i]
            made.append(token)
            text = self.tok.decode(made, skip_special_tokens=False)
            if self.grammar is not None:
                gtext += self.grammar.strings[i]
            if token == self.tok.eos_token_id or ACTION_END in text or (
                    text.count(";") >= STEPS and "<" in text.rsplit(";", 1)[-1]) or (
                    self.grammar is not None and action_prefix_state(gtext) == "done"):
                break
            logits = dec.step(token)[0]
        fed = len(made) - 1
        dec.store(self.cache)
        self.kv_len += fed
        self.next_pos += fed
        held = self.tok.decode(made[:-1], skip_special_tokens=False)
        if self.keep_transcript:
            self.transcript.append(held)
        if ACTION_END not in text and "<" in text.rsplit(";", 1)[-1]:
            text = text[: text.rfind("<")] + ACTION_END
        elif ACTION_END not in text:
            text = text + ACTION_END  # ran out of tokens: close it so the template stays well-formed
        self.owed = text[len(held):] if text.startswith(held) else ACTION_END
        return text, fed


def torch_cat_keep(x, head: int, tail: int):
    """Keep the first `head` and last `tail` positions of a (b, h, L, d) KV tensor."""
    import torch

    return torch.cat([x[:, :, :head], x[:, :, -tail:]], dim=2).contiguous()


# -- execution through robonet ------------------------------------------------------------


class RobonetExecutor:
    """Plays an Action through a robonet RobonetAIClient: 6 steps of 33 ms,
    keys pressed/released on change, the chunk's mouse movement split evenly
    over the steps as relative moves (the pointer moves *by* that much), and
    clicks where the pointer is. In-game Minecraft re-centres the pointer and
    reads each move as camera rotation (needs rawMouseInput:false); on menus
    the moves add up and move the cursor, as with a real mouse. The frame
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
            self.client.send_mouse_button(MOUSE_BUTTONS[token], pressed, at_current=True)
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
                self.client.send_mouse_move_rel(dx, dy)
            time.sleep(max(0.0, self.step_s - (time.perf_counter() - t0)))

    def release_all(self) -> None:
        self._set_held([])


class PipelinedExecutor(RobonetExecutor):
    """Plays chunks on its own thread, back to back, while the caller
    computes the next one. If the next chunk is late, the last step's keys
    stay held (a step lasts until the next one begins). Records, per chunk,
    when its frame was captured and when its first step ran."""

    def __init__(self, client, step_s: float = 1.0 / 30.0):
        super().__init__(client, step_s)
        import queue
        import threading

        self.queue: "queue.Queue" = queue.Queue()
        self.chunk_end = time.perf_counter()  # when the chunk being played finishes
        self.timings: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, action: Action, frame_time: float) -> None:
        self.queue.put((action, frame_time))

    def _run(self) -> None:
        import queue

        while not self._stop.is_set():
            try:
                action, frame_time = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            start = time.perf_counter()
            self.chunk_end = start + STEPS * self.step_s
            prev = self.timings[-1]["start"] if self.timings else None
            self.timings.append({"frame_time": frame_time, "start": start,
                                 "reaction_s": start - frame_time,
                                 "interval_s": None if prev is None else start - prev})
            self.execute(action)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.release_all()


def reasoning_probe(model, processor, system_prompt: str, frame, max_new_tokens: int = 384,
                    temperature: float = 1.0, history=()) -> dict:
    """Evaluation only: let the model think before acting on one frame (the
    model card's reasoning mode, prefill "<think>"), to read what it attends
    to. history: past (frame, action content) turns, as the actor keeps them.
    Returns the thought, the action text after it, and whether it parsed."""
    import torch

    pol = GameScalingPolicy.__new__(GameScalingPolicy)
    pol.system, pol.prefill, pol.history_len = system_prompt, PREFILL_REASONING, len(history)
    pol.turns = list(history)
    msgs = pol.messages(frame)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                         continue_final_message=True)
    inputs = processor(text=[text], images=[f for f, _ in history] + [frame], return_tensors="pt").to(model.device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None, use_cache=True)
    finally:
        model.train(was_training)
    completion = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    thought, _, after = completion.partition("</think>")
    action = parse_action(PREFILL_REASONING + completion)
    return {"thought": thought.strip(), "after": after.strip()[:200], "parsed": action.parsed,
            "finished_thinking": "</think>" in completion, "tokens": int(out.shape[1] - inputs["input_ids"].shape[1])}
