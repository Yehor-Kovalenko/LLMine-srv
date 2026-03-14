"""
GGUF Embedder Engine

Loads a GGUF model with embedding support via llama-cpp-python.

llama.cpp has native embedding mode (``embedding=True``) which runs the
forward pass and returns the pooled hidden state without autoregressive
sampling.  Models such as nomic-embed-text-v1.5.Q4_K_M.gguf work this way.
"""

import asyncio
import logging
from typing import Any

import numpy as np

from ._base_engine import BaseEngine
from src.backend.model_registry_dto import ModelPackage

logger = logging.getLogger(__name__)


class GGUFEmbedEngine(BaseEngine):
    """
    Embedding engine backed by llama-cpp-python in embedding mode.

    Expected input_data keys:
        input       (str | list[str]) – text(s) to embed
        normalize   (bool)            – L2-normalise output, default True
    """

    def __init__(
        self,
        model_package: ModelPackage,
        n_gpu_layers: int = 0,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._verbose = verbose
        super().__init__(model_package)

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def load_logic(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for the GGUF engine. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        path = self.package.path
        n_ctx = self._n_ctx or self.package.context_length
        logger.info(
            "Loading GGUF embedder from '%s'  n_ctx=%d  n_gpu_layers=%d",
            path, n_ctx, self._n_gpu_layers,
        )

        model = Llama(
            model_path=path,
            embedding=True,       # ← critical: disables KV cache, enables embed mode
            n_ctx=n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )
        logger.info("GGUF embedder loaded: %s", self.package.id)
        return model

    async def generate(self, input_data: dict) -> dict:
        texts = input_data.get("input", "")
        if isinstance(texts, str):
            texts = [texts]

        normalize: bool = input_data.get("normalize", True)

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embed(texts, normalize),
        )

        return {
            "embeddings": embeddings,
            "model_id": self.package.id,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    async def generate_stream(self, input_data: dict) -> Any:
        """Embedders do not stream; delegate to generate."""
        return await self.generate(input_data)

    async def free(self) -> None:
        logger.info("Freeing GGUF embedder '%s'", self.package.id)
        if hasattr(self.model, "close"):
            self.model.close()
        del self.model
        self.model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str], normalize: bool) -> list[list[float]]:
        """
        llama-cpp-python's ``embed()`` method accepts a single string.
        We call it once per text to keep it simple; for high-throughput use
        the batch API introduced in llama-cpp-python ≥ 0.2.56.
        """
        all_embeddings: list[list[float]] = []

        for text in texts:
            # embed() returns list[float] directly
            vec = self.model.embed(text)
            arr = np.array(vec, dtype=np.float32)

            if normalize:
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    arr = arr / norm

            all_embeddings.append(arr.tolist())

        return all_embeddings