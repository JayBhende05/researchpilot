import json

from researchpilot.generation.prompt import build_prompt
from researchpilot.generation.schemas import (
    Citation,
    GeneratedAnswer,
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

        if not chunks:

            return RAGResponse(
                question=question,
                answer=(
                    "I could not find relevant information "
                    "in the available documents."
                ),
                citations=[],
            )

        # ----------------------------------------
        # 2. Build prompt
        # ----------------------------------------

        prompt = build_prompt(
            question=question,
            chunks=chunks,
        )

        # ----------------------------------------
        # 3. Generate structured answer
        # ----------------------------------------

        raw_response = self.llm.generate(
            prompt
        )

        try:

            generated = GeneratedAnswer.model_validate(
                json.loads(raw_response)
            )

        except Exception as exc:

            raise ValueError(
                "LLM returned invalid structured output"
            ) from exc

        # ----------------------------------------
        # 4. Build lookup of valid sources
        # ----------------------------------------

        chunk_lookup = {
            chunk["id"]: chunk
            for chunk in chunks
        }

        # ----------------------------------------
        # 5. Validate citations
        # ----------------------------------------

        valid_citation_ids = []

        for citation_id in generated.citation_ids:

            if citation_id in chunk_lookup:

                valid_citation_ids.append(
                    citation_id
                )

        # ----------------------------------------
        # 6. Build citation objects
        # ----------------------------------------

        citations = []

        for citation_id in valid_citation_ids:

            chunk = chunk_lookup[citation_id]

            metadata = chunk.get(
                "metadata",
                {},
            )

            citations.append(
                Citation(
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
            )

        # ----------------------------------------
        # 7. Return final response
        # ----------------------------------------

        return RAGResponse(
            question=question,
            answer=generated.answer,
            citations=citations,
        )
