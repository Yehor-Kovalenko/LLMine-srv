# import ollama
from typing import Any

from src.backend.model_registry import BaseProvider


class OllamaProvider(BaseProvider):

    def load(self, name: str):
        return {"model": name}

    def free(self, model: Any) -> None:
        pass