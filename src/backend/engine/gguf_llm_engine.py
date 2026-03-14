"""
GGUF LLM Engine  (backed by ctransformers)

Loads a GGUF-quantised language model via ctransformers — a zero-build-dependency
alternative to llama-cpp-python that ships pre-built wheels for Windows, macOS,
and Linux.

    - Model is instantiated via AutoModelForCausalLM.from_pretrained()
    - Inference is called via model(prompt) which returns a str or str generator
    - context_length / gpu_layers are set at load time in a config dict
    - repetition_penalty (not repeat_penalty) is the kwarg name
    - No usage/token-count metadata in responses
"""

import asyncio
import logging
import threading
from typing import Any, AsyncGenerator

from ._base_engine import BaseEngine
from src.backend.model_registry_dto import ModelPackage

logger = logging.getLogger(__name__)


# ctransformers model_type strings for common architectures.
# If the architecture is not in this map, we pass it raw and let
# ctransformers raise a descriptive error.
_ARCH_TO_CT_TYPE: dict[str, str] = {
    "llama":    "llama",
    "mistral":  "mistral",
    "falcon":   "falcon",
    "mpt":      "mpt",
    "starcoder":"starcoder",
    "gptj":     "gptj",
    "gpt2":     "gpt2",
    "gpt-neox": "gpt_neox",
    "dolly":    "dolly-v2",
    "replit":   "replit",
    "qwen":     "qwen",
    "phi":      "phi-msft",
    "gemma":    "gemma",
    "deepseek": "llama",
}


class GGUFLLMEngine(BaseEngine):
    """
    Text generation engine backed by ctransformers (GGUF format).

    Expected input_data keys:
        prompt             (str)        – user message
        system_prompt      (str)        – optional system prefix
        temperature        (float)      – default 0.7
        max_new_tokens     (int)        – default 512
        top_p              (float)      – default 0.9
        top_k              (int)        – default 50
        repetition_penalty (float)      – default 1.1
        stop               (list[str])  – optional stop sequences
    """

    def __init__(
        self,
        model_package: ModelPackage,
        gpu_layers: int = 0,
        context_length: int | None = None,
        n_threads: int | None = None,
    ):
        """
        Args:
            gpu_layers:      Layers to offload to GPU (0 = CPU only, -1 = all).
            context_length:  Override the context window from ModelPackage.
            n_threads:       CPU thread count. None → ctransformers default.
        """
        self._gpu_layers     = gpu_layers
        self._context_length = context_length
        self._n_threads      = n_threads
        super().__init__(model_package)

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def load_logic(self):
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "ctransformers is required for the GGUF engine.\n"
                "Install it with:  pip install ctransformers\n"
                "GPU (CUDA):       pip install ctransformers[cuda]"
            ) from exc

        path = self.package.path
        model_type = self._resolve_model_type()
        context_length = self._context_length or self.package.context_length

        config: dict = {
            "context_length": context_length,
            "gpu_layers":     self._gpu_layers,
        }
        if self._n_threads is not None:
            config["threads"] = self._n_threads

        logger.info(
            "Loading GGUF LLM from '%s'  model_type=%s  context_length=%d  gpu_layers=%d",
            path, model_type, context_length, self._gpu_layers,
        )

        model = AutoModelForCausalLM.from_pretrained(
            path,
            model_type=model_type,
            model_file=path if path.endswith(".gguf") else None,
            config=config,
            local_files_only=True,
        )

        logger.info("GGUF LLM loaded: %s", self.package.id)
        return model

    async def generate(self, input_data: dict) -> dict:
        prompt = self._build_prompt(input_data)
        kwargs = self._build_kwargs(input_data)

        loop = asyncio.get_event_loop()
        # ctransformers generation is synchronous; run in a thread pool
        text = await loop.run_in_executor(
            None,
            lambda: self.model(prompt, stream=False, **kwargs),
        )

        return {
            "text": text.strip(),
            "model_id": self.package.id,
        }

    async def generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        prompt = self._build_prompt(input_data)
        kwargs = self._build_kwargs(input_data)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _run() -> None:
            try:
                # ctransformers yields plain str chunks when stream=True
                for chunk in self.model(prompt, stream=True, **kwargs):
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        thread.join()

    async def free(self) -> None:
        logger.info("Freeing GGUF LLM '%s'", self.package.id)
        # Releasing the Python object triggers __del__ in ctransformers,
        # which frees the underlying C++ model and its memory.
        del self.model
        self.model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_type(self) -> str:
        """
        Map the architecture string from ModelPackage to a ctransformers
        model_type.  Falls back to the raw value so ctransformers can raise
        its own descriptive error if the type is genuinely unknown.
        """
        arch = self.package.architecture.lower()
        for key, ct_type in _ARCH_TO_CT_TYPE.items():
            if key in arch:
                return ct_type
        logger.warning(
            "Unknown architecture '%s'; passing raw value to ctransformers.", arch
        )
        return arch or "llama"

    @staticmethod
    def _build_prompt(input_data: dict) -> str:
        system_prompt: str = input_data.get("system_prompt", "")
        prompt: str        = input_data.get("prompt", "")

        if system_prompt:
            return (
                f"### System:\n{system_prompt}\n\n"
                f"### User:\n{prompt}\n\n"
                f"### Assistant:\n"
            )
        return prompt

    @staticmethod
    def _build_kwargs(input_data: dict) -> dict:
        kwargs: dict = dict(
            max_new_tokens     = int(input_data.get("max_new_tokens", 512)),
            temperature        = float(input_data.get("temperature", 0.7)),
            top_p              = float(input_data.get("top_p", 0.9)),
            top_k              = int(input_data.get("top_k", 50)),
            repetition_penalty = float(input_data.get("repetition_penalty", 1.1)),
        )
        stop = input_data.get("stop")
        if stop:
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]
        return kwargs