from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkingConfig:

    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    size_unit: str = "characters"
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    preserve_metadata: bool = True


@dataclass(slots=True)
class Chunk:
    """
    Standard representation of a text chunk.
    """

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """
    Interface for all chunking strategies.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """
        Split text into chunks.
        """
        raise NotImplementedError


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Simple character-based chunking strategy.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        step = self.chunk_size - self.chunk_overlap

        while start < text_length:

            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step

        return chunks


class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    Splits text using progressively smaller separators.

    Priority:
        paragraphs ↓ lines ↓ sentences ↓ words ↓ characters
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            ", ",
            " ",
            "",
        ]

    def chunk(self, text: str) -> list[str]:

        return self._split_recursive(
            text=text,
            separators=self.separators,
        )

    def _split_recursive(self,text: str,separators: list[str],) -> list[str]:

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = separators[0]

        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        chunks: list[str] = []
        current = ""

        for part in parts:

            candidate = f"{current}{separator}{part}" if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            if len(part) > self.chunk_size and len(separators) > 1:

                nested_chunks = self._split_recursive(
                    part,
                    separators[1:],
                )

                chunks.extend(nested_chunks)
                current = ""

            else:
                current = part

        if current:
            chunks.append(current.strip())

        return self._apply_overlap(chunks)

    def _apply_overlap(self,chunks: list[str],) -> list[str]:

        if self.chunk_overlap == 0:
            return chunks

        result: list[str] = []

        for index, chunk in enumerate(chunks):

            if index == 0:
                result.append(chunk)
                continue

            previous = chunks[index - 1]

            overlap = previous[-self.chunk_overlap :]

            result.append(f"{overlap} {chunk}".strip())

        return result

    def chunk(self, text: str) -> list[str]:
        """
        Split text into semantically meaningful chunks.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        chunks = self.splitter.split_text(text)

        if self.min_chunk_size is not None:
            chunks = [
                chunk for chunk in chunks if len(chunk.strip()) >= self.min_chunk_size
            ]

        return [chunk.strip() for chunk in chunks if chunk.strip()]


class LangChainRecursiveStrategy(ChunkingStrategy):
    """
    Adapter around LangChain's recursive splitter.
    """

    def __init__(self,chunk_size: int = 1000,chunk_overlap: int = 200,):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str) -> list[str]:

        return self.splitter.split_text(text)


class ChunkerService:
    """
    Application-level chunking service.

    The service does not know how chunking is performed.
    It delegates chunking to a strategy.
    """

    def __init__(self,strategy: ChunkingStrategy,):
        self.strategy = strategy

    def chunk(self,text: str,*,metadata: dict[str, Any] | None = None,) -> list[Chunk]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        raw_chunks = self.strategy.chunk(text)

        base_metadata = metadata or {}

        return [
            Chunk(
                text=chunk,
                index=index,
                metadata={
                    **base_metadata,
                    "chunk_index": index,
                    "chunk_size": len(chunk),
                },
            )
            for index, chunk in enumerate(raw_chunks)
        ]
