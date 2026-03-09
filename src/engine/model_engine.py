import logging
from typing import Any, AsyncIterator

import asyncio

from ._base import _LazyEngine, MODEL_PATH

logger = logging.getLogger(__name__)

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
    logger.info("load_model(%s) — backend=%s", name)

    raise NotImplementedError("Install transformers/torch and uncomment above.")

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
