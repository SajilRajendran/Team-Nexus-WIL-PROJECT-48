"""Stage 2 - Retrieve: vector search + BM25 keyword search, merged with Reciprocal
Rank Fusion, optionally re-ranked by a cross-encoder, with an abstention signal."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import config
from .embed_index import get_client

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by can do does for from has have how i in is it its of on
or that the this to was what when where which who why will with""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    vector_score: float | None = None   # cosine similarity (1 = identical)
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    found_by: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def citation(self) -> dict:
        m = self.metadata
        page = m.get("page", -1)
        return {
            "title": m.get("title", ""),
            "url": m.get("url", ""),
            "activity_centre": m.get("activity_centre", ""),
            "publisher": m.get("publisher", ""),
            "page": page if page and page > 0 else None,
            "doc_id": m.get("doc_id", ""),
        }


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    max_similarity: float
    top_rerank: float | None
    abstain: bool
    reason: str = ""


class Retriever:
    def __init__(self, embedder, collection=None, use_hybrid: bool = config.USE_HYBRID,
                 use_reranker: bool = config.USE_RERANKER, reranker=None):
        self.embedder = embedder
        self.collection = collection or get_client().get_collection(config.COLLECTION_NAME)
        indexed_with = (self.collection.metadata or {}).get("embedder")
        if indexed_with and indexed_with != embedder.model_name:
            raise RuntimeError(
                f"Index was built with '{indexed_with}' but the current embedder is "
                f"'{embedder.model_name}'. Rebuild with `python -m src.embed_index`.")
        self.use_hybrid = use_hybrid
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._store: dict[str, tuple[str, dict]] = {}
        if use_hybrid:
            self._build_bm25()
        self.reranker = reranker
        if use_reranker and reranker is None:
            self.reranker = _load_reranker()

    # ------------------------------------------------------------------ BM25
    def _build_bm25(self):
        # BM25Plus keeps a positive IDF even on small corpora (BM25Okapi can zero it out)
        from rank_bm25 import BM25Plus

        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = list(data["ids"])
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            self._store[cid] = (doc, meta)
        # include the title so "VicSmart guide" style queries hit the right doc
        corpus = [tokenize(f"{m.get('title', '')} {d}") for d, m in
                  zip(data["documents"], data["metadatas"])]
        self._bm25_vocab = [set(toks) for toks in corpus]
        self._bm25 = BM25Plus(corpus) if corpus else None

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        qset = set(q)
        # BM25Plus gives every doc a small positive score, so require a real term match
        matched = [i for i in np.argsort(scores)[::-1] if qset & self._bm25_vocab[i]]
        return [(self._bm25_ids[i], float(scores[i])) for i in matched[:k]]

    # ---------------------------------------------------------------- vector
    def _vector_search(self, query: str, k: int) -> list[RetrievedChunk]:
        q = self.embedder.embed_query(query)
        n = min(k, self.collection.count())
        if n == 0:
            return []
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=n,
                                    include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append(RetrievedChunk(cid, doc, meta, vector_score=1.0 - float(dist),
                                      found_by=["vector"]))
        return out

    # ------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int | None = None,
                 candidate_k: int | None = None,
                 min_similarity: float | None = None,
                 min_rerank: float | None = None) -> RetrievalResult:
        # read config at call time so settings can be changed at runtime / in tests
        top_k = top_k or config.TOP_K
        candidate_k = candidate_k or config.CANDIDATE_K
        min_similarity = config.MIN_SIMILARITY if min_similarity is None else min_similarity
        min_rerank = config.MIN_RERANK_SCORE if min_rerank is None else min_rerank
        vec_hits = self._vector_search(query, candidate_k)
        max_sim = max((c.vector_score for c in vec_hits), default=0.0)
        merged: dict[str, RetrievedChunk] = {}

        for rank, c in enumerate(vec_hits):
            c.rrf_score = 1.0 / (60 + rank + 1)
            merged[c.chunk_id] = c

        if self.use_hybrid:
            for rank, (cid, score) in enumerate(self._bm25_search(query, candidate_k)):
                if cid in merged:
                    item = merged[cid]
                    item.found_by.append("bm25")
                else:
                    doc, meta = self._store[cid]
                    item = merged[cid] = RetrievedChunk(cid, doc, meta, found_by=["bm25"])
                item.bm25_score = score
                item.rrf_score += 1.0 / (60 + rank + 1)

        candidates = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
        candidates = candidates[:candidate_k]

        top_rerank = None
        if self.reranker and candidates:
            scores = self.reranker.predict([(query, c.text) for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            top_rerank = candidates[0].rerank_score

        final = candidates[:top_k]
        abstain, reason = False, ""
        if not final:
            abstain, reason = True, "no documents retrieved"
        elif max_sim < min_similarity:
            abstain, reason = True, f"best similarity {max_sim:.2f} < {min_similarity}"
        elif top_rerank is not None and top_rerank < min_rerank:
            abstain, reason = True, f"best rerank score {top_rerank:.2f} < {min_rerank}"
        return RetrievalResult(final, max_sim, top_rerank, abstain, reason)


def _load_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(config.RERANKER_MODEL)
    except Exception as exc:
        print(f"[retriever] Re-ranker unavailable ({exc}); continuing without it.")
        return None
