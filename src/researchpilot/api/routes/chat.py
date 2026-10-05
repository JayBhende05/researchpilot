import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from researchpilot.api.deps import (
    Components,
    TimedLLM,
    TimedRetriever,
    get_components,
)
from researchpilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    LatencyMs,
)
from researchpilot.pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# Plain `def` so FastAPI runs the blocking work in its threadpool.
@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    components: Components = Depends(get_components),
) -> ChatResponse:

    timings = {"retrieval": 0.0, "generation": 0.0}

    # RAGPipeline only holds two references, so a per-request
    # instance keeps the timing state request-local.
    pipeline = RAGPipeline(
        retriever=TimedRetriever(components.retriever, timings),
        llm=TimedLLM(components.llm, timings),
    )

    start = time.perf_counter()

    try:
        result = pipeline.query(
            question=request.question,
            top_k=5,
            candidate_k=20,
        )
    except Exception:
        logger.exception("RAG pipeline failed")

        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer",
        )

    total = (time.perf_counter() - start) * 1000

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        citations=result.citations,
        latency_ms=LatencyMs(
            retrieval=round(timings["retrieval"], 1),
            generation=round(timings["generation"], 1),
            total=round(total, 1),
        ),
    )
