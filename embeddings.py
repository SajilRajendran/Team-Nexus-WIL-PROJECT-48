"""Embedding models.

* SentenceTransformerEmbedder - the real one (bge-base / MiniLM), runs locally on CPU.
* HashingEmbedder - dependency-light fallback with no model download. It is lexical,
  not semantic, so use it only for unit tests / CI or when the model cannot be downloaded.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config

# bge models expect this prefix on *queries* (not documents) for best retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64)

    def embed_query(self, text: str) -> np.ndarray:
        if self.is_bge:
            text = _BGE_QUERY_PREFIX + text
        return self.model.encode([text], normalize_embeddings=True)[0]


class HashingEmbedder:
    model_name = "hashing-1024"

    def __init__(self, n_features: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=n_features, alternate_sign=False,
                                     ngram_range=(1, 2), stop_words="english", norm="l2")

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


@lru_cache(maxsize=2)
def get_embedder(kind: str | None = None, model_name: str | None = None):
    kind = (kind or config.EMBEDDER).lower()
    if kind == "hash":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name or config.EMBEDDING_MODEL)
    except Exception as exc:  # no internet / package missing
        print(f"[embeddings] Could not load sentence-transformers model ({exc}); "
              "falling back to HashingEmbedder. Results will be lexical only.")
        return HashingEmbedder()
