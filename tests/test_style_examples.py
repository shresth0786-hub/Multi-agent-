"""Tests for the few-shot style examples loader."""

import json
from pathlib import Path

from app.services import style_examples


def _patch_settings(monkeypatch, tmp_path):
    class FakeSettings:
        style_examples_file = tmp_path / "style_examples.jsonl"

    monkeypatch.setattr(style_examples, "get_settings", lambda: FakeSettings())
    style_examples.get_style_examples.cache_clear()


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def test_parses_single_line_json(monkeypatch, tmp_path):
    path = tmp_path / "style_examples.jsonl"
    _write(path, [
        '{"question": "q1", "answer": "a1"}',
        '"bad line"',
        '{"question": "q2", "answer": "a2"}',
    ])
    _patch_settings(monkeypatch, tmp_path)
    examples = style_examples.get_style_examples()
    assert [e["question"] for e in examples] == ["q1", "q2"]


def test_parses_pretty_printed_json_and_comments(monkeypatch, tmp_path):
    path = tmp_path / "style_examples.jsonl"
    _write(path, [
        "# a comment",
        '{"question": "q1",',
        ' "answer": "a1"}',
        '{"question": "q2", "answer": "a2"}',
    ])
    _patch_settings(monkeypatch, tmp_path)
    examples = style_examples.get_style_examples()
    assert len(examples) == 2
    assert examples[0]["answer"] == "a1"


def test_empty_when_file_missing(monkeypatch, tmp_path):
    _patch_settings(monkeypatch, tmp_path)
    assert style_examples.get_style_examples() == []
    assert style_examples.style_prompt_section() == ""


def test_prompt_section_contains_examples_and_is_bounded(monkeypatch, tmp_path):
    path = tmp_path / "style_examples.jsonl"
    big = "x" * 5000
    _write(path, [
        json.dumps({"question": "q1", "answer": big}),
        json.dumps({"question": "q2", "answer": "answer two"}),
    ])
    _patch_settings(monkeypatch, tmp_path)
    section = style_examples.style_prompt_section()
    assert "## STYLE EXAMPLES TO IMITATE" in section
    assert "answer two" in section
    assert len(section) < 3000
    assert "x" * (style_examples.ANSWER_CHARS + 1) not in section
