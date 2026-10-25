"""Score the chatbot against tests/eval_questions.csv.

Usage:
    python -m eval.run_eval                     # full run (retrieval + generation)
    python -m eval.run_eval --repeat 3          # repeat each question for stable latency
    python -m eval.run_eval --sweep             # retrieval-only threshold sweep for abstention
    python -m eval.run_eval --backend extractive  # baseline without the LLM

Outputs (eval/results/):
    results_<stamp>.csv  - one row per question, with blank columns for manual grading
    summary_<stamp>.md   - the results table for your report

Metrics
    retrieval_hit   expected source doc_id appears in the top-k retrieved chunks
    auto_overlap    share of key words from expected_answer found in the answer (rough
                    proxy only - use the manual 1-5 correctness grades in the report)
    abstention      unanswerable questions where the bot correctly said "I don't know"
    false_abstain   answerable questions where the bot wrongly said "I don't know"
    latency         end-to-end seconds per question (mean and p95)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.retriever import tokenize  # noqa: E402

QUESTIONS = ROOT / "tests" / "eval_questions.csv"
OUT_DIR = ROOT / "eval" / "results"
MANUAL_COLS = ["correctness_rater1_1to5", "correctness_rater2_1to5", "faithful_yes_no",
               "grader_notes"]


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["category"] = r["category"].strip().lower()
        r["expected_sources"] = [s.strip() for s in r.get("expected_source", "").split(";") if s.strip()]
    return rows


def key_overlap(expected: str, answer: str) -> float | None:
    exp = set(tokenize(expected))
    if not exp:
        return None
    return round(len(exp & set(tokenize(answer))) / len(exp), 2)


def p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def summarise(rows: list[dict]) -> list[dict]:
    table = []
    for cat in ["known", "inferred", "unanswerable", "ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        answerable = [r for r in rs if r["category"] != "unanswerable"]
        unans = [r for r in rs if r["category"] == "unanswerable"]
        lat = [float(r["latency_s"]) for r in rs]
        ov = [r["auto_overlap"] for r in answerable if r["auto_overlap"] not in ("", None)]
        table.append({
            "category": cat,
            "n": len(rs),
            "retrieval_hit_%": _pct([r["retrieval_hit"] for r in answerable if r["expected_sources"]]),
            "auto_overlap_avg": round(statistics.mean(ov), 2) if ov else "",
            "false_abstain_%": _pct([r["abstained"] for r in answerable]),
            "abstention_%": _pct([r["abstained"] for r in unans]),
            "latency_mean_s": round(statistics.mean(lat), 2),
            "latency_p95_s": round(p95(lat), 2),
        })
    return table


def _pct(flags: list[bool]) -> str:
    return f"{100 * sum(flags) / len(flags):.0f}" if flags else ""


def write_outputs(rows: list[dict], table: list[dict], meta: dict) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"results_{stamp}.csv"
    cols = ["id", "category", "question", "expected_answer", "expected_source", "answer",
            "abstained", "retrieval_hit", "retrieved_docs", "cited_docs", "auto_overlap",
            "max_similarity", "latency_s"] + MANUAL_COLS
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md_path = OUT_DIR / f"summary_{stamp}.md"
    head = list(table[0].keys())
    lines = [f"# Evaluation summary ({stamp})", "",
             *[f"- **{k}**: {v}" for k, v in meta.items()], "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in table]
    lines += ["", "retrieval_hit_% and auto_overlap apply to answerable questions only; "
              "abstention_% applies to unanswerable questions only. auto_overlap is a rough "
              "automatic proxy - report the manual correctness grades from the CSV."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run(args):
    from src.llm import get_llm
    from src.rag_pipeline import RAGPipeline

    pipe = RAGPipeline(llm=get_llm(args.backend))
    questions = load_questions(Path(args.questions))
    rows = []
    for q in questions:
        lats, a = [], None
        for _ in range(args.repeat):
            a = pipe.ask(q["question"])
            lats.append(a.timings["total_s"])
        retrieved_docs = list(dict.fromkeys(r["doc_id"] for r in a.retrieved))
        rows.append({
            **q,
            "answer": a.answer,
            "abstained": a.abstained,
            "retrieval_hit": bool(set(q["expected_sources"]) & set(retrieved_docs)),
            "retrieved_docs": ";".join(retrieved_docs),
            "cited_docs": ";".join(dict.fromkeys(s["doc_id"] for s in a.sources)),
            "auto_overlap": "" if a.abstained else key_overlap(q["expected_answer"], a.answer),
            "max_similarity": round(a.debug["max_similarity"], 3),
            "latency_s": round(statistics.mean(lats), 2),
        })
        flag = "ABSTAIN" if a.abstained else ("hit" if rows[-1]["retrieval_hit"] else "miss")
        print(f"  {q['id']:<5} {q['category']:<12} {flag:<8} {rows[-1]['latency_s']:>5}s  "
              f"{q['question'][:60]}")

    table = summarise(rows)
    meta = {"llm": pipe.llm.name + (f" ({config.OLLAMA_MODEL})" if pipe.llm.name == "ollama" else ""),
            "embedder": pipe.retriever.embedder.model_name,
            "hybrid": pipe.retriever.use_hybrid,
            "reranker": pipe.retriever.reranker is not None,
            "top_k": config.TOP_K, "min_similarity": config.MIN_SIMILARITY,
            "repeats_per_question": args.repeat}
    csv_path, md_path = write_outputs(rows, table, meta)
    print("\n" + md_path.read_text())
    print(f"Per-question results (grade these manually): {csv_path}")


def sweep(args):
    """Retrieval-only: how does the abstention threshold trade off correct abstentions
    against wrongly refusing answerable questions? Pick the knee of this curve."""
    from src.embeddings import get_embedder
    from src.retriever import Retriever

    retr = Retriever(get_embedder())
    questions = load_questions(Path(args.questions))
    sims = []
    for q in questions:
        res = retr.retrieve(q["question"], min_similarity=-1, min_rerank=-1e9)
        sims.append((q["category"], res.max_similarity, res.top_rerank))
    print(f"{'threshold':>9} | {'correct abstain %':>17} | {'false abstain %':>15}")
    for t in [x / 100 for x in range(20, 86, 5)]:
        un = [s for c, s, _ in sims if c == "unanswerable"]
        an = [s for c, s, _ in sims if c != "unanswerable"]
        ca = 100 * sum(s < t for s in un) / max(len(un), 1)
        fa = 100 * sum(s < t for s in an) / max(len(an), 1)
        print(f"{t:>9.2f} | {ca:>17.0f} | {fa:>15.0f}")
    print("\nPer question max similarity:")
    for q, (c, s, rr) in zip(questions, sims):
        rr_s = f"  rerank={rr:.2f}" if rr is not None else ""
        print(f"  {q['id']:<5} {c:<12} sim={s:.3f}{rr_s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--backend", choices=["ollama", "extractive"], default=None)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    sweep(args) if args.sweep else run(args)


if __name__ == "__main__":
    main()
