from typing import Any, AsyncIterator

import torch

from .base_model_registry import BaseProvider


class HuggingFaceProvider(BaseProvider):

    def stream(self, model: Any, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
               stop: list[str] | None) -> AsyncIterator[str]:
        pass

    def free(self, model: Any) -> None:
        pass

    def load(self, name):

        tokenizer = AutoTokenizer.from_pretrained(name)

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        return {
            "model": model,
            "tokenizer": tokenizer
        }