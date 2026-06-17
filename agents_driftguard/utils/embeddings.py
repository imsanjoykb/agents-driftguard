from __future__ import annotations
import hashlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))

class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

class DummyEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Expand the 64-char hex digest to fill `dim` slots
        raw: list[float] = []
        for i in range(self.dim):
            # Cycle through hex chars, map 0-15 → [-1, 1]
            hex_val = int(digest[i % len(digest)], 16)
            raw.append((hex_val / 15.0) * 2 - 1)
        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None  # noqa: ANN401

    def _load_model(self) -> Any:  # noqa: ANN401
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for this backend. "
                "Install it with: pip install agents-driftguard[embeddings]"
            ) from exc
        return SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            self._model = self._load_model()
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: Any | None = None,  # noqa: ANN401
    ) -> None:
        self.model = model
        self._client = client

    def _get_client(self) -> Any:  # noqa: ANN401
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for this backend. "
                "Install it with: pip install agents-driftguard[openai]"
            ) from exc
        self._client = OpenAI()
        return self._client

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

from typing import Any 


def get_embedder(backend: str = "dummy") -> EmbeddingBackend:
    _backends: dict[str, type[EmbeddingBackend]] = {
        "dummy": DummyEmbeddingBackend,
        "sentence-transformers": SentenceTransformerBackend,
        "openai": OpenAIEmbeddingBackend,
    }
    cls = _backends.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            f"Choose from: {', '.join(_backends.keys())}"
        )
    return cls()
