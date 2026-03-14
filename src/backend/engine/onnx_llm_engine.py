"""
ONNX LLM Engine

Runs a causal language model exported to ONNX format via the Optimum library.
Generation is implemented as a manual token-by-token loop so that streaming
works without a separate thread.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

from transformers import AutoTokenizer

from src.backend.model_registry_dto import ModelPackage
from ._base_engine import BaseEngine

logger = logging.getLogger(__name__)


class ONNXLLMEngine(BaseEngine):
    """
    Text generation engine backed by an ONNX-exported causal LM.

    Uses ``optimum.onnxruntime.ORTModelForCausalLM`` (or Seq2Seq) which wraps
    ONNX Runtime with an interface compatible with HuggingFace ``generate()``.

    Expected input_data keys:
        prompt         (str)   – user message
        system_prompt  (str)   – optional system prefix
        temperature    (float) – default 0.7
        max_new_tokens (int)   – default 512
        top_p          (float) – default 0.9
        top_k          (int)   – default 50
        repetition_penalty (float) – default 1.0
    """

    def __init__(self, model_package: ModelPackage, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self.tokenizer = None
        super().__init__(model_package)

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def load_logic(self):
        model_path = Path(self.package.path)
        # Optimum expects the directory that contains the .onnx file
        model_dir = model_path if model_path.is_dir() else model_path.parent

        logger.info("Loading ONNX LLM from '%s' (gpu=%s)", model_dir, self._use_gpu)

        provider = "CUDAExecutionProvider" if self._use_gpu else "CPUExecutionProvider"

        # Import here so the rest of the codebase doesn't hard-depend on optimum
        try:
            from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
        except ImportError as exc:
            raise ImportError(
                "optimum[onnxruntime] is required for the ONNX LLM engine. "
                "Install it with: pip install 'optimum[onnxruntime]'"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            model = ORTModelForCausalLM.from_pretrained(
                model_dir,
                provider=provider,
                use_io_binding=self._use_gpu,
            )
        except Exception:
            logger.debug("ORTModelForCausalLM failed, trying ORTModelForSeq2SeqLM")
            model = ORTModelForSeq2SeqLM.from_pretrained(
                model_dir,
                provider=provider,
            )

        logger.info("ONNX LLM loaded from '%s'", model_dir)
        return model

    async def generate(self, input_data: dict) -> dict:
        input_ids = self._build_input_ids(input_data)
        gen_kwargs = self._build_gen_kwargs(input_data, input_ids)

        loop = asyncio.get_event_loop()
        output_ids = await loop.run_in_executor(
            None,
            lambda: self.model.generate(**gen_kwargs),
        )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"text": text.strip(), "model_id": self.package.id}

    async def generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        """
        Token-by-token greedy/sampling loop that yields each decoded token as
        a string chunk, giving a streaming feel without a background thread.

        Note: this is a simplified loop – for full beam search / sampling parity
        with ``generate()``, use the non-streaming path.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        input_ids = self._build_input_ids(input_data)
        gen_kwargs = self._build_gen_kwargs(input_data, input_ids)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        for chunk in streamer:
            yield chunk
            await asyncio.sleep(0)

        thread.join()

    async def free(self) -> None:
        logger.info("Freeing ONNX LLM session for '%s'", self.package.id)
        # ORTModel holds an InferenceSession; releasing the Python object
        # is sufficient for ONNX Runtime to free its native resources.
        del self.model
        self.model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_input_ids(self, input_data: dict):
        system_prompt: str = input_data.get("system_prompt", "")
        prompt: str = input_data.get("prompt", "")

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt

        return self.tokenizer(full_prompt, return_tensors="pt").input_ids

    def _build_gen_kwargs(self, input_data: dict, input_ids) -> dict:
        return dict(
            input_ids=input_ids,
            max_new_tokens=int(input_data.get("max_new_tokens", 512)),
            temperature=float(input_data.get("temperature", 0.7)),
            top_p=float(input_data.get("top_p", 0.9)),
            top_k=int(input_data.get("top_k", 50)),
            repetition_penalty=float(input_data.get("repetition_penalty", 1.0)),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )