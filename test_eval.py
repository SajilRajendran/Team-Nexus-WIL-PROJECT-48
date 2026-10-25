from eval.run_eval import QUESTIONS, key_overlap, load_questions, summarise


def test_eval_set_is_well_formed():
    qs = load_questions(QUESTIONS)
    cats = {q["category"] for q in qs}
    assert cats == {"known", "inferred", "unanswerable"}
    assert len({q["id"] for q in qs}) == len(qs), "duplicate ids"
    for q in qs:
        if q["category"] == "unanswerable":
            assert not q["expected_answer"]
        else:
            assert q["expected_answer"] and q["expected_sources"]


def test_key_overlap():
    assert key_overlap("10 business days", "It takes 10 business days [1].") == 1.0
    assert key_overlap("", "anything") is None


def test_summary_metrics():
    rows = [
        {"category": "known", "expected_sources": ["a"], "retrieval_hit": True,
         "abstained": False, "auto_overlap": 1.0, "latency_s": 2},
        {"category": "known", "expected_sources": ["a"], "retrieval_hit": False,
         "abstained": True, "auto_overlap": "", "latency_s": 4},
        {"category": "unanswerable", "expected_sources": [], "retrieval_hit": False,
         "abstained": True, "auto_overlap": "", "latency_s": 1},
    ]
    t = {r["category"]: r for r in summarise(rows)}
    assert t["known"]["retrieval_hit_%"] == "50"
    assert t["known"]["false_abstain_%"] == "50"
    assert t["unanswerable"]["abstention_%"] == "100"
    assert t["ALL"]["n"] == 3
