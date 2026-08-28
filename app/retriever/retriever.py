from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):
    """
    Abstract interface for document retrieval.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """
        Retrieve the most relevant documents for a query.
        """
        raise NotImplementedError
