"""Local LoRA adapter loader for models/lora_adapter/ (Llama-3-8B + PEFT)."""

from functools import lru_cache
from pathlib import Path
from typing import Any


class _LocalLoraChatWrapper:
    """Minimal LangChain-compatible chat wrapper around HF + PEFT pipeline."""

    def __init__(self, pipe, tokenizer, max_new_tokens: int = 512):
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _format_messages(self, messages) -> str:
        # Accept LangChain messages or plain string
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            role = getattr(m, "type", getattr(m, "role", "user"))
            content = getattr(m, "content", str(m))
            if isinstance(content, list):
                content = " ".join([x.get("text", "") if isinstance(x, dict) else str(x) for x in content])
            if role in {"human", "user"}:
                parts.append(f"User: {content}")
            elif role in {"ai", "assistant"}:
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")
            else:
                parts.append(str(content))
        parts.append("Assistant:")
        return "\n".join(parts)

    def invoke(self, input, *args, **kwargs):
        from langchain_core.messages import AIMessage

        prompt = self._format_messages(input)
        # HF pipeline with PEFT already merged
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.2, pad_token_id=self.tokenizer.eos_token_id)
        text = out[0]["generated_text"]
        # Strip prompt
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return AIMessage(content=text)

    def stream(self, *args, **kwargs):
        # Fallback: non-streaming
        yield self.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.pipe, name, None)


@lru_cache(maxsize=1)
def _load_pipeline(adapter_path: str, base_model: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
    import torch

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}")

    # Tokenizer from base
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model - 4-bit if bitsandbytes available, else fp16 on CPU
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load base model {base_model}. "
            f"For Meta-Llama you need HF_TOKEN and `huggingface-cli login`. "
            f"Error: {e}"
        ) from e

    # Merge LoRA
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return pipe, tokenizer


def get_lora_chat_model(adapter_path: str, base_model: str, temperature: float = 0.2, max_new_tokens: int = 512) -> Any:
    pipe, tokenizer = _load_pipeline(adapter_path, base_model)
    # temperature arg kept for compat, pipeline uses 0.2
    return _LocalLoraChatWrapper(pipe, tokenizer, max_new_tokens=max_new_tokens)
