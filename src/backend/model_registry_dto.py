from dataclasses import dataclass
from enum import Enum


class ModelFormat(Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"


@dataclass
class ModelPackage:
    path: str # path to the downloaded model
    format: ModelFormat # format of the model
    model_type: str # general type (classifier, llm, embedder etc.)
    metadata: dict #additional useful metadata