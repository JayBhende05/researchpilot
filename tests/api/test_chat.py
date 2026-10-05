import json

from conftest import CHUNKS, make_client

URL = "/api/v1/chat"


def test_chat_happy_path(client):
    response = client.post(URL, json={"question": "What is it?"})

    assert response.status_code == 200

    body = response.json()

    assert body["question"] == "What is it?"
    assert body["answer"] == "It works."
    assert [c["chunk_id"] for c in body["citations"]] == [CHUNKS[0]["id"]]
    assert body["citations"][0]["document_id"] == "paper.pdf"

    latency = body["latency_ms"]

    assert set(latency) == {"retrieval", "generation", "total"}
    assert latency["total"] >= latency["retrieval"] + latency["generation"]


def test_unknown_citation_ids_are_dropped():
    llm_response = json.dumps(
        {
            "answer": "x",
            "citation_ids": [CHUNKS[1]["id"], "made_up_id"],
        }
    )

    with make_client(llm_response=llm_response) as client:
        body = client.post(URL, json={"question": "q"}).json()

    assert [c["chunk_id"] for c in body["citations"]] == [CHUNKS[1]["id"]]


def test_empty_retrieval_returns_canned_answer():
    with make_client(chunks=[]) as client:
        response = client.post(URL, json={"question": "q"})

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert "could not find" in response.json()["answer"]


def test_invalid_question(client):
    assert client.post(URL, json={}).status_code == 422
    assert client.post(URL, json={"question": "   "}).status_code == 422
    assert client.post(URL, json={"question": "a" * 1001}).status_code == 422


def test_llm_returns_non_json():
    with make_client(llm_response="not json") as client:
        response = client.post(URL, json={"question": "q"})

    assert response.status_code == 502
