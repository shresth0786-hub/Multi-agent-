"""
Fine-tune the local LoRA adapter (models/lora_adapter) on
nvidia/Nemotron-SFT-Agentic-v4 (agentic SFT, ~1M samples).

Uses HF datasets + transformers + peft + trl SFTTrainer.
Base model: meta-llama/Meta-Llama-3-8B (from adapter_config.json) + existing LoRA.

All storage forced to D: drive (HF_HOME, datasets, checkpoints).

Run:
  hf auth login  # paste HF token with access to Llama-3 and Nemotron dataset
  python -m scripts.train_nemotron --dataset nvidia/Nemotron-SFT-Agentic-v4 --max_steps 100 --output_dir D:\\Multi Agent Ai\\models\\lora_adapter_nemotron_v4

Requires: datasets, transformers, peft, trl, accelerate, torch
"""

import argparse
import os
from pathlib import Path

# Force all HF/datasets cache to D: drive (C: only 940MB free)
os.environ.setdefault("HF_HOME", r"D:\Multi Agent Ai\.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Multi Agent Ai\.hf_cache\hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\Multi Agent Ai\.hf_cache\transformers")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\Multi Agent Ai\.hf_cache\datasets")
os.environ.setdefault("TRL_CACHE", r"D:\Multi Agent Ai\.hf_cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "lora_adapter"
DEFAULT_OUTPUT = Path(r"D:\Multi Agent Ai\models\lora_adapter_nemotron_v4")
DEFAULT_DATASET = "nvidia/Nemotron-SFT-Agentic-v4"


def load_nemotron_dataset(dataset: str = DEFAULT_DATASET, token: str | None = None, streaming: bool = False, max_samples: int | None = None):
    from datasets import load_dataset

    # Gated - needs HF_TOKEN via env or `hf auth login`
    ds = load_dataset(
        dataset,
        split="train",
        streaming=streaming,
        token=token,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    if max_samples and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_nemotron(example):
    # Dataset schema: check common fields - fallback to generic
    # Expected: instruction / input / output / messages / prompt / response
    # We normalize to "text" for SFTTrainer
    if "messages" in example and isinstance(example["messages"], list):
        # Chat format: list[{role, content}]
        text = ""
        for m in example["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
        text = text.strip()
        return {"text": text}
    # Try instruction/output
    instr = example.get("instruction") or example.get("prompt") or example.get("input") or ""
    out = example.get("output") or example.get("response") or example.get("answer") or ""
    if instr and out:
        return {"text": f"User: {instr}\nAssistant: {out}"}
    # Fallback: join all string fields
    text = " ".join([str(v) for v in example.values() if isinstance(v, str)])[:2000]
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id, e.g. nvidia/Nemotron-SFT-Agentic-v4")
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=200, help="Cap dataset for quick demo; 0 = full")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset (no download, for large)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    print(f"Dataset: {args.dataset}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Base: {args.base_model}")
    print(f"Output: {args.output_dir} (on D:)")
    print(f"HF cache: {os.environ.get('HF_HOME')}")
    print(f"HF token: {'yes' if token else 'no - run `hf auth login` if dataset is gated'}")

    # 1. Load dataset
    print(f"\n[1] Loading {args.dataset} ...")
    try:
        ds = load_nemotron_dataset(dataset=args.dataset, token=token, streaming=args.streaming, max_samples=args.max_samples if args.max_samples else None)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print(f"Hint: `hf auth login` and ensure token has access to {args.dataset} and {args.base_model}")
        return 1

    # Map to text field
    print(f"Dataset loaded: {type(ds)}")
    if not args.streaming:
        print(f"Samples: {len(ds)}")
        ds = ds.map(format_nemotron, remove_columns=ds.column_names)
        # Filter empty
        ds = ds.filter(lambda x: len(x["text"].strip()) > 20)
        print(f"After formatting: {len(ds)} samples")
        print("Example:", ds[0]["text"][:500])
    else:
        # streaming: map lazily
        ds = ds.map(format_nemotron)

    # 2. Load tokenizer & model + existing LoRA
    print("\n[2] Loading base model + existing LoRA adapter ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=token,
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=True,  # needs bitsandbytes, saves VRAM
    )
    # Load existing adapter if present
    adapter_path = Path(args.adapter_path)
    if (adapter_path / "adapter_model.safetensors").exists():
        print(f"Loading existing adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        print("No existing adapter found, will train from scratch")

    # 3. Train with TRL SFTTrainer
    print("\n[3] Training with SFTTrainer ...")
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig

    # New LoRA config (merge with existing or override)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=50,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    print(f"\n[4] Saving to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Done. To use it, set in .env: LLM_PROVIDER=lora + LORA_ADAPTER_PATH=models/lora_adapter_nemotron")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
