"""Build (or rebuild) the ChromaDB vector index from data/processed/chunks.jsonl.

Usage:
    python -m src.embed_index            # rebuild from scratch
"""
from __future__ import annotations

import time
from pathlib import Path

from . import config
from .embeddings import get_embedder
from .ingest import Chunk, load_chunks


def get_client(chroma_dir: Path = config.CHROMA_DIR):
    import chromadb
    from chromadb.config import Settings

    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir),
                                     settings=Settings(anonymized_telemetry=False))


def build_index(chunks: list[Chunk] | None = None, embedder=None,
                chroma_dir: Path = config.CHROMA_DIR,
                collection_name: str = config.COLLECTION_NAME):
    chunks = chunks if chunks is not None else load_chunks()
    if not chunks:
        raise SystemExit("No chunks to index - run `python -m src.ingest` first.")
    embedder = embedder or get_embedder()
    client = get_client(chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name,
        # we supply our own vectors, so Chroma never downloads its default model
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedder": embedder.model_name},
    )
    t0 = time.perf_counter()
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vectors = embedder.embed_documents([c.text for c in part])
        collection.add(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.metadata() for c in part],
            embeddings=[v.tolist() for v in vectors],
        )
    print(f"Indexed {len(chunks)} chunks with {embedder.model_name} "
          f"in {time.perf_counter() - t0:.1f}s -> {chroma_dir}")
    return collection


if __name__ == "__main__":
    build_index()
