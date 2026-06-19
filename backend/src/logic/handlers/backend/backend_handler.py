from abc import ABC, abstractmethod


class BackendHandler(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...
