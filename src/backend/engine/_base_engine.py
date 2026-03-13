from abc import ABC, abstractmethod
from typing import Any

from src.backend.model_registry_dto import ModelPackage


class BaseEngine(ABC):
    def __init__(self, model_package: ModelPackage):
        self.package = model_package
        self.model = self.load_logic()

    @abstractmethod
    def load_logic(self):
        ...

    @abstractmethod
    async def generate(self, input_data: dict) -> Any:
        ...

    @abstractmethod
    async def generate_stream(self, input_data: dict) -> Any:
        ...