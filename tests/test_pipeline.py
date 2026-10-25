from src import config
from src.llm import NOT_FOUND, build_messages


def test_retrieval_finds_expected_document(retriever):
    res = retriever.retrieve("How many business days does a fast-track permit take?")
    assert not res.abstain
    assert res.chunks[0].metadata["doc_id"] == "exampleton-permits"


def test_hybrid_merges_vector_and_bm25(retriever):
    res = retriever.retrieve("bus interchange Exampleton Station")
    sources = {s for c in res.chunks for s in c.found_by}
    assert {"vector", "bm25"} <= sources


def test_unrelated_question_abstains(retriever):
    res = retriever.retrieve("Who won the 1987 cricket world cup?")
    assert res.abstain


def test_pipeline_answers_with_citation(pipeline):
    a = pipeline.ask("How many storeys are allowed near Exampleton Station?")
    assert not a.abstained
    assert "12 storeys" in a.answer
    assert a.sources and a.sources[0]["url"].startswith("https://example.org")
    assert a.timings["total_s"] >= 0


def test_pipeline_abstains_out_of_scope(pipeline):
    a = pipeline.ask("What is the capital city of Peru?")
    assert a.abstained
    assert a.answer == config.ABSTAIN_MESSAGE
    assert a.sources == []


def test_llm_not_found_triggers_abstention(retriever, embedder):
    from src.rag_pipeline import RAGPipeline

    class RefusingLLM:
        name = "fake"

        def generate(self, *a, **k):
            return NOT_FOUND

    p = RAGPipeline(retriever=retriever, llm=RefusingLLM(), embedder=embedder)
    a = p.ask("How long does a fast-track permit take?")
    assert a.abstained


def test_only_cited_sources_returned(retriever, embedder):
    from src.rag_pipeline import RAGPipeline

    class OneCiteLLM:
        name = "fake"

        def generate(self, *a, **k):
            return "It takes 10 business days [1]."

    p = RAGPipeline(retriever=retriever, llm=OneCiteLLM(), embedder=embedder)
    a = p.ask("How long does a fast-track permit take?")
    assert [s["ref"] for s in a.sources] == [1]


def test_follow_up_question_uses_history(pipeline):
    from src.rag_pipeline import RAGPipeline

    hist = [{"question": "How tall can buildings be near Exampleton Station?", "answer": "..."}]
    rq = RAGPipeline.retrieval_query("what about homes?", hist)
    assert "Exampleton Station" in rq
    standalone = "How many submissions did the Exampleton consultation receive overall?"
    assert RAGPipeline.retrieval_query(standalone, hist) == standalone


def test_prompt_contains_rules_and_numbered_context(retriever):
    res = retriever.retrieve("fast-track permit")
    msgs = build_messages("q?", res.chunks)
    assert NOT_FOUND in msgs[0]["content"]
    assert "[1] Source:" in msgs[-1]["content"]
