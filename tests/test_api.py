import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(pipeline):
    from src.api import create_app

    return TestClient(create_app(pipeline))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "How many homes will the Exampleton rezoning support?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"answer", "abstained", "sources", "timings"}
    assert "4,500" in body["answer"]
    assert body["sources"][0]["title"]


def test_ask_abstains_out_of_scope(client):
    body = client.post("/ask", json={"question": "Best pizza recipe in Naples?"}).json()
    assert body["abstained"] is True


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_debug_mode_returns_retrieved_chunks(client):
    body = client.post("/ask", json={"question": "fast-track permit decision time",
                                     "debug": True}).json()
    assert body["debug"]["retrieved"]


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Activity Centres Assistant" in r.text
