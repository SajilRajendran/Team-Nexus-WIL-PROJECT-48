"""Shared fixtures: a tiny index built from FICTIONAL documents in tests/fixtures,
using the offline HashingEmbedder and the ExtractiveLLM, so tests need no downloads,
no GPU and no Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_chunks(tmp_path_factory):
    from src.ingest import ingest

    out = tmp_path_factory.mktemp("processed") / "chunks.jsonl"
    return ingest(FIXTURES, out, FIXTURES / "sources.csv")


@pytest.fixture(scope="session")
def embedder():
    from src.embeddings import HashingEmbedder

    return HashingEmbedder()


@pytest.fixture(scope="session")
def collection(fixture_chunks, embedder, tmp_path_factory):
    from src.embed_index import build_index

    return build_index(fixture_chunks, embedder, tmp_path_factory.mktemp("chroma"), "test")


@pytest.fixture(scope="session")
def retriever(collection, embedder):
    from src.retriever import Retriever

    return Retriever(embedder, collection, use_hybrid=True, use_reranker=False)


@pytest.fixture(scope="session")
def pipeline(retriever, embedder):
    from src.llm import ExtractiveLLM
    from src.rag_pipeline import RAGPipeline

    return RAGPipeline(retriever=retriever, llm=ExtractiveLLM(), embedder=embedder)


@pytest.fixture(autouse=True)
def low_threshold(monkeypatch):
    """The hashing embedder gives lower cosine scores than a semantic model."""
    from src import config

    monkeypatch.setattr(config, "MIN_SIMILARITY", 0.15)
