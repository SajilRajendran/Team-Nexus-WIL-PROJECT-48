"""FastAPI app: POST /ask, GET /health, and the chat page at /.

Run:
    uvicorn src.api:app --reload --port 8000
then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config

FRONTEND = config.ROOT / "frontend" / "index.html"


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=10)
    debug: bool = False


class Source(BaseModel):
    ref: int
    title: str
    url: str = ""
    activity_centre: str = ""
    publisher: str = ""
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    timings: dict
    debug: dict | None = None


def create_app(pipeline=None) -> FastAPI:
    app = FastAPI(title="Team Nexus - Activity Centres RAG chatbot", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    state = {"pipeline": pipeline}

    def get_pipeline():
        if state["pipeline"] is None:  # lazy load so /health works instantly
            from .rag_pipeline import RAGPipeline
            state["pipeline"] = RAGPipeline()
        return state["pipeline"]

    @app.get("/health")
    def health():
        p = state["pipeline"]
        return {"status": "ok", "loaded": p is not None,
                "llm": getattr(getattr(p, "llm", None), "name", None)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            a = get_pipeline().ask(req.question, [t.model_dump() for t in req.history])
        except Exception as exc:  # surface a clean error to the frontend
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
        return AskResponse(answer=a.answer, abstained=a.abstained,
                           sources=[Source(**s) for s in a.sources], timings=a.timings,
                           debug={**a.debug, "retrieved": a.retrieved} if req.debug else None)

    @app.get("/")
    def index():
        if not Path(FRONTEND).exists():
            raise HTTPException(404, "frontend/index.html not found")
        return FileResponse(FRONTEND)

    return app


app = create_app()
