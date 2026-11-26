"""GameScaling actor: prompt, action parsing and robonet execution (no GPU)."""

import os
import re
from pathlib import Path

import pytest

from sili_quartz.gamescaling import (
    KEY_VOCAB, MINECRAFT_KEYS, MINECRAFT_RULES, NOOP_TEXT, TO_ROBONET, RobonetExecutor,
    build_system_prompt, parse_action,
)

MODEL_DIR = Path(os.environ.get("GS_MODEL_DIR", os.path.expanduser("~/models/GameScaling-0.8B-work")))


def _card_doom_example() -> str | None:
    readme = MODEL_DIR / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text()
    m = re.search(r"### Complete example: Doom Battle-1.*?```text\n(.*?)\n```", text, re.DOTALL)
    return m.group(1) if m else None


def test_system_prompt_matches_the_model_cards_doom_example_byte_for_byte():
    card = _card_doom_example()
    if card is None:
        pytest.skip(f"no model card at {MODEL_DIR}")
    rules = card.split("**Game Rules**\n", 1)[1].split("\n**Common Keys in", 1)[0]
    keys = [("W", "Move forward"), ("A", "Strafe left"), ("S", "Move backward"), ("D", "Strafe right"),
            ("LB", "Fire weapon"), ("Ctrl", "Fire weapon (keyboard alternative)"), ("Shift", "Run"),
            ("Space", "Open doors / use switches"), ("one", "Select weapon slot 1"),
            ("two", "Select weapon slot 2"), ("three", "Select weapon slot 3")]
    assert build_system_prompt("Doom Battle-1", rules, keys) == card


def test_minecraft_prompt_uses_only_training_key_names_and_states_no_goal():
    prompt = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    assert all(k in KEY_VOCAB for k, _ in MINECRAFT_KEYS)
    assert "**Common Keys in Minecraft**" in prompt and prompt.count("Minecraft") >= 2
    for word in ("goal", "score", "task", "objective"):
        assert word not in MINECRAFT_RULES.lower()


def test_parses_the_model_cards_reply():
    a = parse_action("<think>\n\n</think>\n\n<|action_start|>-37 -3 0 ; LB S ; LB ; LB ; LB ; LB ; LB<|action_end|>")
    assert a.parsed and not a.truncated
    assert a.mouse == (-37, -3, 0)
    assert a.steps == [["LB", "S"], ["LB"], ["LB"], ["LB"], ["LB"], ["LB"]]


def test_unclosed_think_is_a_truncated_noop():
    a = parse_action("<think>I should look left and <|action_start|>1 2 3 ; W")
    assert a.truncated and not a.parsed and a.text == NOOP_TEXT


def test_reads_the_action_after_the_last_think_close_only():
    a = parse_action("<think>format: <|action_start|>9 9 0 ; Q ; ; ; ; ;<|action_end|></think>"
                     "<|action_start|>5 0 0 ; W ; W ; ; ; ;<|action_end|>")
    assert a.mouse == (5, 0, 0) and a.steps[0] == ["W"]


def test_unknown_keys_are_dropped_groups_capped_and_values_clipped():
    a = parse_action("<|action_start|>2000 -5000 9 ; W Foo A S D Space ; ;<|action_end|>")
    assert a.parsed and a.mouse == (1000, -1000, 5)
    assert a.steps[0] == ["W", "A", "S", "D"]  # at most 4
    assert a.dropped == ["Foo"]
    assert len(a.steps) == 6 and a.steps[2:] == [[], [], [], []]  # missing groups are empty


def test_bare_fallback_and_garbage():
    assert parse_action("3 4 0 ; W ; ; ; ; ;").parsed
    assert not parse_action("I like trees").parsed
    assert not parse_action("<|action_start|>a b c ; W<|action_end|>").parsed


def test_history_text_round_trips():
    a = parse_action("<|action_start|>-12 3 0 ; W Space ; W ; ; ; ; LB<|action_end|>")
    b = parse_action(a.text)
    assert (b.mouse, b.steps) == (a.mouse, a.steps)


def test_every_vocab_key_maps_to_robonet_except_the_ones_it_lacks():
    unmapped = {k for k in KEY_VOCAB if k not in TO_ROBONET and k not in ("LB", "RB", "MB")}
    assert unmapped == {"Alt", "Pause", "Caps"}


class _FakeClient:
    def __init__(self):
        self.calls = []

    def send_key(self, key, pressed):
        self.calls.append(("key", key, pressed))

    def send_mouse_button(self, button, pressed, at_current=False):
        assert at_current  # clicks land where the pointer is; nothing re-anchors it
        self.calls.append(("button", button, pressed))

    def send_mouse_move_rel(self, dx, dy):
        self.calls.append(("move", round(dx, 4), round(dy, 4)))

    def send_scroll(self, delta):
        self.calls.append(("scroll", delta))


def test_executor_presses_on_change_and_moves_the_mouse_relatively_over_six_steps():
    c = _FakeClient()
    ex = RobonetExecutor(c, step_s=0.0)
    ex.execute(parse_action("<|action_start|>60 -30 0 ; W ; W LB ; LB ; ; ; Ctrl<|action_end|>"))
    keys = [x for x in c.calls if x[0] != "move"]
    assert keys == [
        ("key", "w", True),
        ("button", "left", True),
        ("key", "w", False),
        ("button", "left", False),
        ("key", "ctrlleft", True),
    ]
    moves = [x for x in c.calls if x[0] == "move"]
    assert moves == [("move", 0.01, -0.005)] * 6  # by 60/1000/6 right and 30/1000/6 up, each step
    ex.release_all()
    assert c.calls[-1] == ("key", "ctrlleft", False)


def test_action_vocab_covers_every_token_of_valid_actions():
    tok_dir = MODEL_DIR / "tokenizer.json"
    if not tok_dir.exists():
        pytest.skip(f"no tokenizer at {MODEL_DIR}")
    from transformers import AutoTokenizer

    from sili_quartz.gamescaling import action_vocab

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab = set(action_vocab(tok))
    for text in ("-37 -3 0 ; LB S ; LB ; LB ; LB ; LB ; LB<|action_end|>",
                 "1000 -1000 5 ; W A S D ; Space Ctrl ; ; ; ; nine<|action_end|>",
                 "0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>"):
        assert set(tok.encode(text, add_special_tokens=False)) <= vocab, text
