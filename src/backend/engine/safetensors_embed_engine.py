"""
Safetensors Embedder Engine

Loads a sentence / token embedding model from a safetensors snapshot using
HuggingFace Transformers and returns dense vector embeddings.
"""

import asyncio
import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ._base_engine import BaseEngine
from src.backend.model_registry_dto import ModelPackage

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SafetensorsEmbedEngine(BaseEngine):
    """
    Embedding engine backed by HuggingFace Transformers safetensors weights.

    Supports single strings and batches.  Embeddings are mean-pooled over the
    last hidden state and L2-normalised by default.

    Expected input_data keys:
        input          (str | list[str]) – text(s) to embed
        normalize      (bool)            – L2-normalise output, default True
        batch_size     (int)             – internal batch size for large inputs, default 32
    """

    def __init__(self, model_package: ModelPackage, device: str | None = None):
        self._device = device or _pick_device()
        self.tokenizer = None
        super().__init__(model_package)

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def load_logic(self):
        path = self.package.path
        logger.info("Loading safetensors embedder from '%s' on device '%s'", path, self._device)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            device_map=self._device,
        )
        model.eval()
        logger.info("Safetensors embedder loaded")
        return model

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
            "embeddings": embeddings,           # list[list[float]]
            "model_id": self.package.id,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    async def generate_stream(self, input_data: dict) -> Any:
        """Embedders do not stream; delegate to generate."""
        return await self.generate(input_data)

    async def free(self) -> None:
        logger.info("Freeing safetensors embedder '%s'", self.package.id)
        del self.model
        if self._device == "cuda":
            torch.cuda.empty_cache()
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
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean-pool over the sequence dimension, respecting the attention mask
            hidden = outputs.last_hidden_state          # (B, T, D)
            mask = encoded["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

            all_embeddings.extend(pooled.cpu().float().tolist())

        return all_embeddings