"""
Factory class that decides what model loader to use depending on the format in which the model was saved and the format of the model (classiffier, llm, embedder etc.)
"""
import logging

from .gguf_embed_engine import GGUFEmbedEngine
from .gguf_llm_engine import GGUFLLMEngine
from .onnx_embed_engine import ONNXEmbedEngine
from .onnx_llm_engine import ONNXLLMEngine
from .safetensors_embed_engine import SafetensorsEmbedEngine
from .safetensors_llm_engine import SafetensorsLLMEngine
from ..model_registry_dto import ModelFormat, ModelPackage

logger = logging.getLogger(__name__)

class EngineFactory:
    _engines = {
        (ModelFormat.SAFETENSORS, "llm"): SafetensorsLLMEngine,
        (ModelFormat.SAFETENSORS, "embed"): SafetensorsEmbedEngine,
        (ModelFormat.GGUF, "llm"): GGUFLLMEngine,
        (ModelFormat.GGUF, "embed"): GGUFEmbedEngine,
        (ModelFormat.ONNX, "llm"): ONNXLLMEngine,
        (ModelFormat.ONNX, "embed"): ONNXEmbedEngine,
    }

    @classmethod
    def create_engine(cls, package: ModelPackage):
        key = (package.format, package.model_type)
        engine_cls = cls._engines.get(key)

        if not engine_cls:
            logger.warning(f"Engine {key} is not supported. Failing back to llm engine.")
            engine_cls = cls._engines.get((ModelFormat.SAFETENSORS, "llm"))

        return engine_cls(package)
