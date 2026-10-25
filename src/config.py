"""Central configuration. Every value can be overridden with an environment variable,
so teammates can tune settings without editing code (e.g. `set RAG_TOP_K=5`)."""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("RAG_DATA_DIR", str(ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_CSV = DATA_DIR / "sources.csv"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = Path(_env("RAG_CHROMA_DIR", str(DATA_DIR / "chroma")))
COLLECTION_NAME = _env("RAG_COLLECTION", "activity_centres")

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE_WORDS = _env("RAG_CHUNK_SIZE", 300)      # ~400 tokens
CHUNK_OVERLAP_WORDS = _env("RAG_CHUNK_OVERLAP", 60)
MIN_CHUNK_WORDS = _env("RAG_MIN_CHUNK_WORDS", 25)

# --- Embeddings -------------------------------------------------------------
# "sentence-transformers" (real semantic model) or "hash" (offline fallback for tests/CI)
EMBEDDER = _env("RAG_EMBEDDER", "sentence-transformers")
EMBEDDING_MODEL = _env("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# --- Retrieval --------------------------------------------------------------
TOP_K = _env("RAG_TOP_K", 3)                 # chunks passed to the LLM
CANDIDATE_K = _env("RAG_CANDIDATE_K", 10)    # chunks fetched before re-ranking
USE_HYBRID = _env("RAG_USE_HYBRID", True)    # vector + BM25 merged with RRF
USE_RERANKER = _env("RAG_USE_RERANKER", True)
RERANKER_MODEL = _env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Abstention: if the best vector similarity is below this, answer "I don't know".
# Tune it on your eval set (see eval/run_eval.py --sweep).
MIN_SIMILARITY = _env("RAG_MIN_SIMILARITY", 0.45)
# Cross-encoder logit threshold (only used when the reranker is active)
MIN_RERANK_SCORE = _env("RAG_MIN_RERANK_SCORE", 0.0)

# --- LLM (Ollama) -----------------------------------------------------------
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = _env("RAG_TEMPERATURE", 0.1)
LLM_TIMEOUT = _env("RAG_LLM_TIMEOUT", 120)
# "ollama" or "extractive" (no LLM - returns the best matching sentences; useful when Ollama is down)
LLM_BACKEND = _env("RAG_LLM_BACKEND", "ollama")

# --- Conversation memory ----------------------------------------------------
HISTORY_TURNS = _env("RAG_HISTORY_TURNS", 2)

ABSTAIN_MESSAGE = (
    "I couldn't find this in the source documents I have, so I don't want to guess. "
    "Try rephrasing, or check the official Planning Victoria or council website."
)
