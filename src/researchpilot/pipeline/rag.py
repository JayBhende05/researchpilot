from researchpilot.generation.prompt import build_prompt
from researchpilot.generation.schemas import (
    Citation,
    RAGResponse,
)


class RAGPipeline:

    def __init__(
        self,
        retriever,
        llm,
    ):
        self.retriever = retriever
        self.llm = llm

    def query(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> RAGResponse:

        # ----------------------------------------
        # 1. Retrieve + rerank
        # ----------------------------------------

        chunks = self.retriever.search(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        # ----------------------------------------
        # 2. Build prompt
        # ----------------------------------------

        prompt = build_prompt(
            question=question,
            chunks=chunks,
        )

        # ----------------------------------------
        # 3. Generate answer
        # ----------------------------------------

        answer = self.llm.generate(prompt)

        # ----------------------------------------
        # 4. Build citations
        # ----------------------------------------

        citations = []

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            citation = Citation(
                chunk_id=chunk["id"],
                document_id=metadata.get(
                    "source",
                    "unknown",
                ),
                text=chunk["document"],
                score=chunk.get(
                    "reranker_score"
                ),
                metadata=metadata,
            )

            citations.append(citation)

        # ----------------------------------------
        # 5. Return structured result
        # ----------------------------------------

        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
        )
