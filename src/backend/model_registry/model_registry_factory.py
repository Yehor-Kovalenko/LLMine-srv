from anyio.functools import lru_cache

from .base_model_registry import BaseProvider
from .hf_model_registry import HFSafetensorsProvider
from .local_model_registry import LocalProvider
from .ollama_model_registry import OllamaProvider

@lru_cache()
def get_llm_provider(kind: str) -> BaseProvider:

    if kind == "hf":
        return HFSafetensorsProvider()
    elif kind == "ollama":
        return OllamaProvider()
    elif kind == "local":
        return LocalProvider()
    else:
        raise ValueError(f"Unknown backend {kind}")