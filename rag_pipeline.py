"""Stage 4 - Respond: glue retrieval + generation + citations together.

CLI usage:
    python -m src.rag_pipeline "How long does a VicSmart permit take?"
    python -m src.rag_pipeline          # interactive chat
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict, dataclass, field

from . import config
from .embeddings import get_embedder
from .llm import NOT_FOUND, get_llm
from .retriever import Retriever, tokenize

_CITE = re.compile(r"\[(\d+)\]")
_FOLLOW_UP_HINTS = {"it", "that", "there", "they", "those", "this", "instead", "same", "also"}


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, llm=None, embedder=None):
        embedder = embedder or get_embedder()
        self.retriever = retriever or Retriever(embedder)
        self.llm = llm or get_llm()

    @staticmethod
    def retrieval_query(question: str, history: list[dict] | None) -> str:
        """Short follow-ups like 'what about Box Hill instead?' lack context on their own,
        so we prepend the previous question for retrieval only."""
        if not history:
            return question
        toks = question.lower().split()
        if len(tokenize(question)) <= 6 or _FOLLOW_UP_HINTS & set(toks):
            return f"{history[-1]['question']} {question}"
        return question

    def ask(self, question: str, history: list[dict] | None = None, **retrieve_kwargs) -> Answer:
        question = question.strip()
        t0 = time.perf_counter()
        rq = self.retrieval_query(question, history)
        result = self.retriever.retrieve(rq, **retrieve_kwargs)
        t1 = time.perf_counter()
        retrieved = [{**c.citation(), "text": c.text, "vector_score": c.vector_score,
                      "rerank_score": c.rerank_score, "found_by": c.found_by}
                     for c in result.chunks]
        debug = {"retrieval_query": rq, "max_similarity": result.max_similarity,
                 "top_rerank": result.top_rerank, "abstain_reason": result.reason,
                 "llm": self.llm.name}

        if result.abstain:
            return Answer(question, config.ABSTAIN_MESSAGE, True, [], retrieved,
                          {"retrieval_s": t1 - t0, "generation_s": 0.0, "total_s": t1 - t0},
                          debug)

        raw = self.llm.generate(question, result.chunks, history)
        t2 = time.perf_counter()
        timings = {"retrieval_s": t1 - t0, "generation_s": t2 - t1, "total_s": t2 - t0}

        if NOT_FOUND in raw or not raw.strip():
            debug["abstain_reason"] = "LLM said the context does not contain the answer"
            return Answer(question, config.ABSTAIN_MESSAGE, True, [], retrieved, timings, debug)

        cited = sorted({int(n) for n in _CITE.findall(raw) if 1 <= int(n) <= len(result.chunks)})
        if not cited:  # model forgot to cite - fall back to all retrieved passages
            cited = list(range(1, len(result.chunks) + 1))
        sources, seen = [], set()
        for n in cited:
            cit = result.chunks[n - 1].citation()
            key = (cit["doc_id"], cit["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"ref": n, **cit})
        return Answer(question, raw, False, sources, retrieved, timings, debug)


def _print_answer(a: Answer):
    print(f"\n{a.answer}\n")
    for s in a.sources:
        page = f", p.{s['page']}" if s["page"] else ""
        print(f"  [{s['ref']}] {s['title']}{page}  {s['url']}")
    print(f"  ({a.timings['total_s']:.1f}s, llm={a.debug['llm']})\n")


def main():
    pipe = RAGPipeline()
    if len(sys.argv) > 1:
        _print_answer(pipe.ask(" ".join(sys.argv[1:])))
        return
    history: list[dict] = []
    print("Team Nexus Activity Centres assistant - type 'exit' to quit.")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        a = pipe.ask(q, history)
        _print_answer(a)
        history.append({"question": q, "answer": a.answer})


if __name__ == "__main__":
    main()
