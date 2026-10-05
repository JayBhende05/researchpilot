import time
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class Components:
    """Heavy, shared objects built once at startup."""

    retriever: object
    llm: object
    chunk_count: int


def build_components() -> Components:
    # Imported here so tests that inject fakes never load the models.
    from researchpilot.generation.llm import GeminiLLM
    from researchpilot.retrieval.bm25 import BM25Retriever
    from researchpilot.retrieval.embedder import Embedder
    from researchpilot.retrieval.reranked import RerankedRetriever
    from researchpilot.retrieval.reranker import Reranker
    from researchpilot.retrieval.vector_store import VectorStore

    vector_store = VectorStore()
    chunks = vector_store.get_all_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks found in ChromaDB. "
            "Run the ingestion pipeline first."
        )

    retriever = RerankedRetriever(
        vector_store=vector_store,
        embedder=Embedder(),
        bm25_retriever=BM25Retriever(chunks=chunks),
        reranker=Reranker(),
    )

    return Components(
        retriever=retriever,
        llm=GeminiLLM(),
        chunk_count=len(chunks),
    )


def get_components(request: Request) -> Components:
    components = getattr(request.app.state, "components", None)

    if components is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded",
        )

    return components


class TimedRetriever:
    """Records retrieval time without changing the retriever."""

    def __init__(self, retriever, timings: dict):
        self._retriever = retriever
        self._timings = timings

    def search(self, *args, **kwargs):
        start = time.perf_counter()

        try:
            return self._retriever.search(*args, **kwargs)
        finally:
            self._timings["retrieval"] = (
                time.perf_counter() - start
            ) * 1000


class TimedLLM:
    """Records generation time without changing the LLM."""

    def __init__(self, llm, timings: dict):
        self._llm = llm
        self._timings = timings

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()

        try:
            return self._llm.generate(prompt)
        finally:
            self._timings["generation"] = (
                time.perf_counter() - start
            ) * 1000
