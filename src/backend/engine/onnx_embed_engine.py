"""
ONNX Embedder Engine

Runs an embedding model exported to ONNX via ONNX Runtime directly (no Optimum
required), making it usable in constrained environments where only
``onnxruntime`` is installed.

Falls back to ``optimum.onnxruntime.ORTModelForFeatureExtraction`` if the raw
session approach fails (e.g. multi-file models with external data).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from ._base_engine import BaseEngine
from src.backend.model_registry_dto import ModelPackage

logger = logging.getLogger(__name__)


class ONNXEmbedEngine(BaseEngine):
    """
    Embedding engine backed by ONNX Runtime.

    Expected input_data keys:
        input       (str | list[str]) – text(s) to embed
        normalize   (bool)            – L2-normalise output, default True
        batch_size  (int)             – default 32
    """

    def __init__(self, model_package: ModelPackage, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self.tokenizer = None
        self._use_optimum = False   # set during load_logic if raw session fails
        super().__init__(model_package)

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def load_logic(self):
        model_path = Path(self.package.path)
        model_dir = model_path if model_path.is_dir() else model_path.parent

        logger.info("Loading ONNX embedder from '%s'", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required"
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_gpu
            else ["CPUExecutionProvider"]
        )

        # Try loading the raw session first (works when .onnx is a single file)
        if model_path.is_file() and model_path.suffix == ".onnx":
            session = ort.InferenceSession(str(model_path), providers=providers)
            logger.info("ONNX embedder loaded via raw InferenceSession")
            return session

        # Multiple files / external data → use Optimum as fallback
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            provider = "CUDAExecutionProvider" if self._use_gpu else "CPUExecutionProvider"
            model = ORTModelForFeatureExtraction.from_pretrained(model_dir, provider=provider)
            self._use_optimum = True
            logger.info("ONNX embedder loaded via ORTModelForFeatureExtraction")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ONNX embedder from '{model_path}'. "
                "Try pointing to the specific .onnx file or install optimum[onnxruntime]."
            ) from exc

    async def generate(self, input_data: dict) -> dict:
        texts = input_data.get("input", "")
        if isinstance(texts, str):
            texts = [texts]

        normalize: bool = input_data.get("normalize", True)
        batch_size: int = int(input_data.get("batch_size", 32))

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embed_batched(texts, normalize, batch_size),
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
        logger.info("Freeing ONNX embedder session for '%s'", self.package.id)
        del self.model
        self.model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_batched(
        self,
        texts: list[str],
        normalize: bool,
        batch_size: int,
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.package.context_length,
                return_tensors="np",  # ONNX Runtime works with numpy
            )

            if self._use_optimum:
                import torch
                # ORTModelForFeatureExtraction expects torch tensors
                torch_enc = {k: torch.from_numpy(v) for k, v in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**torch_enc)
                hidden = outputs.last_hidden_state.numpy()
                mask = encoded["attention_mask"][..., np.newaxis].astype(np.float32)
            else:
                # Raw ONNX Runtime session
                input_names = {inp.name for inp in self.model.get_inputs()}
                feed = {k: v for k, v in encoded.items() if k in input_names}
                # Some models also want token_type_ids; add zeros if missing
                if "token_type_ids" in input_names and "token_type_ids" not in feed:
                    feed["token_type_ids"] = np.zeros_like(encoded["input_ids"])
                outputs = self.model.run(None, feed)
                hidden = outputs[0]   # (B, T, D) – last hidden state
                mask = encoded["attention_mask"][..., np.newaxis].astype(np.float32)

            # Mean-pool
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

            if normalize:
                norms = np.linalg.norm(pooled, axis=-1, keepdims=True).clip(min=1e-9)
                pooled = pooled / norms

            all_embeddings.extend(pooled.astype(np.float32).tolist())

        return all_embeddings