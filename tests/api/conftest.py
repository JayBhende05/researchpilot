import json

import pytest
from fastapi.testclient import TestClient

from researchpilot.api.deps import Components
from researchpilot.api.main import create_app

CHUNKS = [
    {
        "id": f"paper.pdf_p1_c{i}",
        "document": f"chunk text {i}",
        "metadata": {"source": "paper.pdf", "page": 1, "chunk": i},
        "reranker_score": 1.0 - i / 10,
    }
    for i in range(3)
]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5, candidate_k=20):
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


def make_client(chunks=CHUNKS, llm_response=None):
    if llm_response is None:
        llm_response = json.dumps(
            {
                "answer": "It works.",
                "citation_ids": [CHUNKS[0]["id"]],
            }
        )

    components = Components(
        retriever=FakeRetriever(chunks),
        llm=FakeLLM(llm_response),
        chunk_count=len(CHUNKS),
    )

    # Context manager runs the app lifespan.
    return TestClient(create_app(components=components))


@pytest.fixture
def client():
    with make_client() as c:
        yield c
