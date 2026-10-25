"""Stage 3 - Generate: prompt assembly + local LLM (Ollama), with an extractive fallback."""
from __future__ import annotations

import re

import requests

from . import config
from .retriever import RetrievedChunk, tokenize

NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = f"""You are a helpful assistant for Victorian residents asking about the
Victorian Government's Activity Centres Program (rezoning, housing density, transport,
planning permits and approval timelines).

Rules:
1. Answer ONLY using the numbered context passages provided. Do not use outside knowledge.
2. Cite the passage(s) you used with square brackets, e.g. [1] or [1][3], after each claim.
3. If the context does not contain the answer, reply with exactly: {NOT_FOUND}
4. Use plain English a resident can understand. Be concise (2-5 sentences or a short list).
5. Never invent numbers, dates, heights, street names or policy names."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        label = m.get("title", "Untitled")
        if m.get("activity_centre"):
            label += f" ({m['activity_centre']})"
        page = m.get("page", -1)
        if page and page > 0:
            label += f", page {page}"
        parts.append(f"[{i}] Source: {label}\n{c.text}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[RetrievedChunk],
                   history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-config.HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content":
                     f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"})
    return messages


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str = config.OLLAMA_MODEL, url: str = config.OLLAMA_URL):
        self.model, self.url = model, url.rstrip("/")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            names = {m["name"] for m in r.json().get("models", [])}
            return r.ok and (self.model in names or f"{self.model}:latest" in names)
        except Exception:
            return False

    def generate(self, question, chunks, history=None) -> str:
        r = requests.post(f"{self.url}/api/chat", timeout=config.LLM_TIMEOUT, json={
            "model": self.model,
            "messages": build_messages(question, chunks, history),
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE, "num_ctx": 4096},
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class ExtractiveLLM:
    """No-LLM fallback: returns the sentences that best overlap with the question.
    Always faithful (it only copies source text) but not fluent. Handy for demos
    when Ollama is unavailable, and as a baseline to compare the LLM against."""
    name = "extractive"

    def available(self) -> bool:
        return True

    def generate(self, question, chunks, history=None, max_sentences: int = 3) -> str:
        q = set(tokenize(question))
        scored = []
        for i, c in enumerate(chunks, start=1):
            for pos, s in enumerate(re.split(r"(?<=[.!?])\s+", c.text)):
                toks = set(tokenize(s))
                if len(toks) < 4:
                    continue
                overlap = len(q & toks) / (len(q) or 1)
                scored.append((overlap, -i, -pos, s.strip(), i))
        scored.sort(reverse=True)
        best = [x for x in scored[:max_sentences] if x[0] > 0]
        if not best:
            return NOT_FOUND
        return " ".join(f"{s} [{i}]" for *_, s, i in best)


def get_llm(backend: str | None = None):
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "extractive":
        return ExtractiveLLM()
    llm = OllamaLLM()
    if llm.available():
        return llm
    print(f"[llm] Ollama model '{config.OLLAMA_MODEL}' not reachable at {config.OLLAMA_URL}. "
          f"Run `ollama pull {config.OLLAMA_MODEL}` and `ollama serve`. "
          "Using the extractive fallback for now.")
    return ExtractiveLLM()
