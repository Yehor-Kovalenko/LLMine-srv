from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from ._base_engine import BaseEngine
from src.backend.model_registry import get_llm_provider
from .engine_factory import EngineFactory
from ..model_registry_dto import ModelPackage

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_TTL_SECONDS: int = int(os.getenv("MODEL_TTL_SECONDS", "3600"))

class LazyLoader:
    """
    Generic lazy-load + TTL eviction engine.
    """

    def __init__(self, provider: str, label: str | None) -> None:
        self._provider = get_llm_provider(provider)
        self._engine: BaseEngine | None = None
        self._model_name: str | None = None
        self._model_package: ModelPackage | None = None

        self._label = label or "None" # user label for the model

        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._eviction_task: asyncio.Task | None = None

    # Lifecycle #####################################
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

    def _load(self, name: str) -> BaseEngine | None:
        package = self._provider.load(name)

        return EngineFactory.create_engine(package)


    async def _eviction_loop(self) -> None:
        check_interval = max(60, MODEL_TTL_SECONDS // 10)
        while True:
            await asyncio.sleep(check_interval)
            async with self._lock:
                if self._engine is not None:
                    idle = time.monotonic() - self._last_used
                    if idle >= MODEL_TTL_SECONDS:
                        logger.info(
                            "[%s] '%s' idle %.0fs — evicting.",
                            self._label, self._model_name, idle,
                        )
                        await self._unload_unsafe()

    async def _unload_unsafe(self) -> None:
        """Must be called with self._lock held."""
        if self._engine is None:
            return
        logger.info("[%s] Unloading '%s'.", self._label, self._model_name)
        # free resources
        self._free()

    async def _unload(self) -> None:
        async with self._lock:
            await self._unload_unsafe()

    def _free(self) -> None:
        """
        Backend specific cleanup
        :return:
        """
        self._provider.free(self._model_name)
        self._engine = None
        self._model_name = None
        self._last_used = 0.0

    # Public API ########################
    async def generate(self, input_data: dict):
        engine = await self.get(self._model_name)
        return await engine.generate(input_data)

    async def generate_stream(self, input_data: dict):
        engine = await self.get(self._model_name)
        return await engine.generate_stream(input_data)


    async def get(self, name: str) -> Any:
        """Return the loaded model, loading lazily if needed."""
        target = name
        # TODO make checking if model name exists
        async with self._lock:
            if self._engine is None or self._model_name != target:
                if self._engine is not None:
                    logger.info(
                        "[%s] Model changed (%s → %s), reloading.",
                        self._label, self._model_name, target,
                    )
                    await self._unload_unsafe()
                logger.info("[%s] Loading '%s' …", self._label, target)
                loop = asyncio.get_running_loop()
                self._engine = await loop.run_in_executor(None, self._load, target)
                self._model_name = target
                logger.info("[%s] '%s' ready.", self._label, target)
            self._last_used = time.monotonic()
        return self._engine

    async def switch(self, name: str) -> None:
        """Evict the current model and schedule the new one for lazy load."""
        async with self._lock:
            if self._engine is not None:
                await self._unload_unsafe()
            # Just update the default; actual load happens on next get() (next request to the model)
        logger.info("[%s] Switched default model to '%s'.", self._label, name)

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def idle_seconds(self) -> float | None:
        if self._last_used == 0.0:
            return None
        return time.monotonic() - self._last_used