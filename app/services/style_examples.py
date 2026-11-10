"""Few-shot style examples loader (question -> ideal answer pairs).

The examples come from ``STYLE_EXAMPLES_FILE`` (default
``data/examples/style_examples.jsonl``), one JSON object per line::

    {"question": "Analyze our WAU retention...", "answer": "## Executive Summary\n..."}

Each line with ``#`` is treated as a comment. At query time the writer prompt
includes a handful of these so the LLM mimics the desired style (zero-cost
"style training"). The same file is the source dataset for ``scripts/finetune``.
"""

import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

MAX_EXAMPLES = 5
QUESTION_CHARS = 220
ANSWER_CHARS = 800


@lru_cache
def get_style_examples() -> list[dict]:
    """Return validated style examples from the configured file."""
    path: Path = get_settings().style_examples_file
    examples: list[dict] = []
    if not path.exists():
        return examples
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "#":  # comment line
            while idx < len(text) and text[idx] != "\n":
                idx += 1
            continue
        if char.isspace():
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict) and obj.get("question") and obj.get("answer"):
            examples.append(obj)
        idx = end
    return examples[:MAX_EXAMPLES]


def style_prompt_section() -> str:
    """Render a compact prompt section the writer should imitate, or empty."""
    examples = get_style_examples()
    if not examples:
        return ""
    blocks = []
    for item in examples:
        q = item["question"].strip()[:QUESTION_CHARS]
        a = item["answer"].strip()[:ANSWER_CHARS]
        blocks.append(f"QUESTION: {q}\nIDEAL ANSWER (mimic this style exactly):\n{a}")
    return (
        "\n\n## STYLE EXAMPLES TO IMITATE\n"
        "Match the structure, tone, formatting and vocabulary of these answers "
        "exactly. Do not mention the examples.\n\n"
        + "\n\n---\n\n".join(blocks)
    )
