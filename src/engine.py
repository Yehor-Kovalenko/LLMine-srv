"""
engine.py — Lazy model loading with TTL-based cache eviction.

Two independent engines:
  - ModelEngine   : text generation (vLLM or HuggingFace)
  - EmbedEngine   : embedding model (HuggingFace sentence-transformers or similar)

Both engines load on the FIRST request (lazy), not at server start.
Both are evicted from memory after MODEL_TTL_SECONDS of inactivity.

Environment variables
---------------------
MODEL_PATH          HF model-id or local path for the inference model
EMBED_MODEL_PATH    HF model-id or local path for the embedder
ENGINE_BACKEND      "vllm" | "huggingface"   (default: huggingface)
MODEL_TTL_SECONDS   Idle seconds before eviction (default: 3600)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_TTL_SECONDS: int = int(os.getenv("MODEL_TTL_SECONDS", "3600"))
ENGINE_BACKEND: str = os.getenv("ENGINE_BACKEND", "huggingface").lower()
MODEL_PATH: str = os.getenv("MODEL_PATH", "gpt2")
EMBED_MODEL_PATH: str = os.getenv("EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")


# ── Stubs: replace with real implementations ──────────────────────────────────

def load_model(name: str) -> Any:
    """
    Load and return the inference model for *name*.

    vLLM example:
        from vllm import LLM
        return LLM(model=name)

    HuggingFace example:
        from transformers import pipeline
        return pipeline("text-generation", model=name, device_map="auto")
    """
    logger.info("load_model(%s) — backend=%s", name, ENGINE_BACKEND)

    if ENGINE_BACKEND == "vllm":
        # from vllm import LLM
        # return LLM(model=name)
        raise NotImplementedError("Install vllm and uncomment the lines above.")

    elif ENGINE_BACKEND == "huggingface":
        # from transformers import pipeline
        # return pipeline("text-generation", model=name, device_map="auto")
        raise NotImplementedError("Install transformers/torch and uncomment above.")

    raise ValueError(f"Unknown ENGINE_BACKEND: {ENGINE_BACKEND!r}")


def load_embedder(name: str) -> Any:
    """
    Load and return the embedding model for *name*.

    sentence-transformers example:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)

    HuggingFace feature-extraction pipeline example:
        from transformers import pipeline
        return pipeline("feature-extraction", model=name)
    """
    logger.info("load_embedder(%s)", name)
    # from sentence_transformers import SentenceTransformer
    # return SentenceTransformer(name)
    raise NotImplementedError("Install sentence-transformers and uncomment above.")


# ── Base lazy engine ──────────────────────────────────────────────────────────

class _LazyEngine:
    """
    Generic lazy-load + TTL eviction engine.
    Subclasses provide _load(name) and use self._model directly.
    """

    def __init__(self, default_name: str, label: str) -> None:
        self._default_name = default_name
        self._label = label
        self._model: Any | None = None
        self._model_name: str | None = None
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._eviction_task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background eviction loop. Model is NOT loaded here."""
        self._eviction_task = asyncio.create_task(
            self._eviction_loop(), name=f"{self._label}-eviction"
        )
        logger.info(
            "[%s] Engine started — lazy load on first request (TTL=%ds)",
            self._label, MODEL_TTL_SECONDS,
        )

    async def stop(self) -> None:
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
        await self._unload()
        logger.info("[%s] Engine stopped.", self._label)

    # ── Public ────────────────────────────────────────────────────────────────

    async def get(self, name: str | None = None) -> Any:
        """Return the loaded model, loading lazily if needed."""
        target = name or self._default_name
        async with self._lock:
            if self._model is None or self._model_name != target:
                if self._model is not None:
                    logger.info(
                        "[%s] Model changed (%s → %s), reloading.",
                        self._label, self._model_name, target,
                    )
                    await self._unload_unsafe()
                logger.info("[%s] Loading '%s' …", self._label, target)
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(None, self._load, target)
                self._model_name = target
                logger.info("[%s] '%s' ready.", self._label, target)
            self._last_used = time.monotonic()
        return self._model

    async def switch(self, name: str) -> None:
        """Evict the current model and schedule the new one for lazy load."""
        async with self._lock:
            if self._model is not None:
                await self._unload_unsafe()
            # Just update the default; actual load happens on next get()
            self._default_name = name
        logger.info("[%s] Switched default model to '%s'.", self._label, name)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def default_name(self) -> str:
        return self._default_name

    @property
    def idle_seconds(self) -> float | None:
        if self._last_used == 0.0:
            return None
        return time.monotonic() - self._last_used

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load(self, name: str) -> Any:
        raise NotImplementedError

    async def _eviction_loop(self) -> None:
        check_interval = max(60, MODEL_TTL_SECONDS // 10)
        while True:
            await asyncio.sleep(check_interval)
            async with self._lock:
                if self._model is not None:
                    idle = time.monotonic() - self._last_used
                    if idle >= MODEL_TTL_SECONDS:
                        logger.info(
                            "[%s] '%s' idle %.0fs — evicting.",
                            self._label, self._model_name, idle,
                        )
                        await self._unload_unsafe()

    async def _unload_unsafe(self) -> None:
        """Must be called with self._lock held."""
        if self._model is None:
            return
        logger.info("[%s] Unloading '%s'.", self._label, self._model_name)
        # Add backend-specific cleanup here if needed, e.g.:
        #   del self._model; torch.cuda.empty_cache()
        self._model = None
        self._model_name = None
        self._last_used = 0.0

    async def _unload(self) -> None:
        async with self._lock:
            await self._unload_unsafe()


# ── Inference engine ──────────────────────────────────────────────────────────

class ModelEngine(_LazyEngine):

    def __init__(self) -> None:
        super().__init__(default_name=MODEL_PATH, label="inference")

    def _load(self, name: str) -> Any:
        return load_model(name)

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        model: str | None = None,
    ) -> tuple[str, int, int]:
        """
        Non-streaming generation.
        Returns (generated_text, prompt_token_count, completion_token_count).

        Wire up your backend here, e.g.:

        vLLM:
            from vllm import SamplingParams
            m = await self.get(model)
            sp = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                top_p=top_p, stop=stop or [])
            out = m.generate([prompt], sp)
            text = out[0].outputs[0].text
            return text, len(out[0].prompt_token_ids), len(out[0].outputs[0].token_ids)

        HuggingFace pipeline:
            m = await self.get(model)
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(
                None,
                lambda: m(prompt, max_new_tokens=max_tokens,
                          temperature=temperature, return_full_text=False)
            )
            text = out[0]["generated_text"]
            return text, len(prompt.split()), len(text.split())
        """
        await self.get(model)  # ensure loaded / touch last_used
        raise NotImplementedError("Implement ModelEngine.generate() for your backend.")

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming generation — yields one token string at a time.

        Wire up your backend here, e.g.:

        vLLM async engine:
            from vllm import AsyncLLMEngine, SamplingParams
            async for output in self._model.generate(prompt, sp, request_id="..."):
                yield output.outputs[0].text

        HuggingFace TextIteratorStreamer:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
            # run model.generate() in a thread, yield from streamer
        """
        await self.get(model)
        # Placeholder — remove and replace with real streaming
        placeholder = "[stream() not yet implemented — wire up your backend]"
        for word in placeholder.split():
            yield word + " "
            await asyncio.sleep(0)


# ── Embedder engine ───────────────────────────────────────────────────────────

class EmbedEngine(_LazyEngine):

    def __init__(self) -> None:
        super().__init__(default_name=EMBED_MODEL_PATH, label="embedder")

    def _load(self, name: str) -> Any:
        return load_embedder(name)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """
        Return a list of embedding vectors, one per input string.

        sentence-transformers example:
            m = await self.get(model)
            loop = asyncio.get_running_loop()
            vecs = await loop.run_in_executor(None, lambda: m.encode(texts).tolist())
            return vecs

        HuggingFace pipeline example:
            m = await self.get(model)
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, lambda: m(texts))
            return [v[0][0] for v in out]   # shape varies by model
        """
        await self.get(model)
        raise NotImplementedError("Implement EmbedEngine.embed() for your backend.")