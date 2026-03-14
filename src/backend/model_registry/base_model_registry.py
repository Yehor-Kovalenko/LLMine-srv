"""
Base class implementing Strategy design pattern. Responsible for defining model backend type a.k.a. model registry from which the model will be loaded
"""
from abc import ABC, abstractmethod

from src.backend.model_registry_dto import ModelPackage


class BaseProvider(ABC):

    @abstractmethod
    def load(self, name: str) -> ModelPackage:
        ...