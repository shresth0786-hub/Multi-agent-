"""Fine-tune an OpenAI model so reports match your example style permanently.

Source dataset: data/examples/style_examples.jsonl  (question -> ideal answer).

Note: answers with NO keys -> Ask: fine-tuning needs OPENAI_API_KEY.
The recommended path is few-shot (automatic, free). Use fine-tuning when you
have at least ~20-50 example pairs and want the style baked into a custom model.

Run:  python -m scripts.finetune
Extra: python -m scripts.finetune --publish
       (writes the fine-tuned model id into .env as OPENAI_MODEL)
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

from app.config import get_settings, usable_key
from app.prompts.writer import WRITER_SYSTEM

BASE_MODEL = "gpt-4o-mini"
SYSTEM_STYLE = (
    "You write structured research reports. Imitate exactly the style, tone, "
    "formatting and vocabulary of the ASSISTANT answer in each example. Do not "
    "mention the examples."
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"


def load_examples(path: Path) -> list[dict]:
    examples: list[dict] = []
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if item.get("question") and item.get("answer"):
            examples.append(item)
    return examples


def build_training_lines(examples: list[dict]) -> list[dict]:
    lines = []
    for ex in examples:
        lines.append(
            {
                "messages": [
                    {"role": "system", "content": WRITER_SYSTEM + "\n\n" + SYSTEM_STYLE},
                    {"role": "user", "content": f"Original request:\n{ex['question']}"},
                    {"role": "assistant", "content": ex["answer"]},
                ]
            }
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune report style.")
    parser.add_argument("--publish", action="store_true", help="Write model id into .env")
    args = parser.parse_args()

    settings = get_settings()
    if not usable_key(settings.openai_api_key):
        print("[!] Fine-tuning needs a real OPENAI_API_KEY in .env.")
        print("    Style training works instantly with few-shot examples and no")
        print("    custom model - see data/examples/style_examples.jsonl.")
        return 1

    examples = load_examples(settings.style_examples_file)
    print(f"Examples loaded: {len(examples)} from {settings.style_examples_file}")
    if len(examples) < 20:
        print(f"[i] You have {len(examples)} - fine-tuning is usually meaningful with 20-50+.")
        print("    For a style match today, few-shot already applies these examples.")

    lines = build_training_lines(examples)
    if not lines:
        print("[!] No usable examples found. Add question/answer JSON lines to the file.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "style_train.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
        encoding="utf-8",
    )
    print(f"Prepared training file: {train_path}")

    print("[1] Uploading training file to OpenAI...")
    client = OpenAI(api_key=settings.openai_api_key)  # noqa: S106 - loaded from .env on purpose
    try:
        uploaded = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")  # noqa: SIM115
    except Exception as exc:
        print(f"    Upload failed: {exc}")
        return 1
    print(f"    File id: {uploaded.id}")

    try:
        job = client.fine_tuning.jobs.create(
            model=BASE_MODEL,
            training_file=uploaded.id,
            suffix="report-style",
        )
        print(f"[2] Fine-tuning job started: {job.id}")

        while True:
            status = client.fine_tuning.jobs.retrieve(job.id)
            print(f"    status: {status.status}", end="\r")
            if status.status in {"succeeded", "failed", "cancelled"}:
                print()
                break
            import time

            time.sleep(20)

        if status.status == "succeeded":
            model_id = status.fine_tuned_model
            print(f"[3] Done. Fine-tuned model: {model_id}")
            if args.publish:
                _write_env_model(model_id)
                print("    OPENAI_MODEL updated in .env - the writer now uses this model.")
            else:
                model_hint = "Set OPENAI_MODEL=<model id> in .env to use it, "
                print(f"    {model_hint}or re-run with --publish.")
            return 0
        print(f"    Job failed: {job.id} - check the OpenAI dashboard.")
        return 1
    except Exception as exc:
        print(f"    Fine-tuning request failed: {exc}")
        return 1


def _write_env_model(model_id: str) -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        env_path.write_text(f"OPENAI_MODEL={model_id}\n", encoding="utf-8")
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_MODEL="):
            lines[i] = f"OPENAI_MODEL={model_id}"
            replaced = True
            break
    if not replaced:
        lines.append(f"OPENAI_MODEL={model_id}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
