"""
Factory class that decides what model loader to use depending on the format in which the model was saved and the format of the model (classiffier, llm, embedder etc.)
"""
from .safetensors_llm_engine import SafetensorsLLMEngine
from ..model_registry_dto import ModelFormat, ModelPackage


class EngineFactory:
    _engines = {
        (ModelFormat.SAFETENSORS, "llm"): SafetensorsLLMEngine
    }

    @classmethod
    def create_engine(cls, package: ModelPackage):
        key = (package.format, package.model_type)
        engine_cls = cls._engines.get(key)

        if not engine_cls:
            raise ValueError(f"Engine {key} is not supported")

        return engine_cls(package)
