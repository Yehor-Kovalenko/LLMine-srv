"""
_base.py — Lazy model loading with TTL-based cache eviction.

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
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_TTL_SECONDS: int = int(os.getenv("MODEL_TTL_SECONDS", "3600"))
MODELS_DIR: str = os.getenv("MODELS_DIR")
EMBED_MODEL_PATH: str = os.getenv("EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")


# Base lazy engine

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

    # Public API

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
            # Just update the default; actual load happens on next get() (next request to the model)
            self._default_name = name
        logger.info("[%s] Switched default model to '%s'.", self._label, name)

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
        # TODO maybe model class with cleanup method
        #   del self._model; torch.cuda.empty_cache()
        self._model = None
        self._model_name = None
        self._last_used = 0.0

    async def _unload(self) -> None:
        async with self._lock:
            await self._unload_unsafe()
