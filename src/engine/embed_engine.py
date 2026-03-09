import logging
from typing import Any

from ._base import _LazyEngine, EMBED_MODEL_PATH

logger = logging.getLogger(__name__)


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