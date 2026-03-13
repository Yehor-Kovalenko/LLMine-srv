from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelFormat(Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"


@dataclass
class ModelPackage:
    id: str # unique model id ("meta/ollama3:4b" for example)
    model_type: str # general type (classifier, llm, embedder etc.)

    path: str # path to the downloaded model
    format: ModelFormat # format of the model

    metadata: dict #additional useful metadata
    architecture: str = "" # architecture llama, bert, mistral etc.
    context_length: int = 2048
    dimensions: Optional[int] = None # dimensions of the embeddings
