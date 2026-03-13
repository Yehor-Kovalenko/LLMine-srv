"""
Base class implementing Strategy design pattern. Responsible for defining model backend type a.k.a. model registry from which the model will be loaded
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):

    @abstractmethod
    def load(self, name: str) -> Any:
        ...

    @abstractmethod
    def free(self, model: Any) -> None:
        ...