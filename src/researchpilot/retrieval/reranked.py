from collections import defaultdict


class RerankedRetriever:
    """
    Hybrid retrieval followed by cross-encoder reranking.

    Pipeline:
        Vector Search
            +
        BM25 Search
            ↓
        RRF Fusion
            ↓
        Candidate Pool
            ↓
        Cross-Encoder Reranker
            ↓
        Final Top-K
    """

    def __init__(
    self,
    vector_store,
    embedder,
    bm25_retriever,
    reranker,
    context_selector,
    rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.context_selector = context_selector
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:

        # --------------------------------------------------
        # Vector retrieval
        # --------------------------------------------------

        query_embedding = self.embedder.embed_query(query)

        vector_results = self.vector_store.search_ranked(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        # --------------------------------------------------
        # RRF fusion
        # --------------------------------------------------

        scores = defaultdict(float)
        result_lookup = {}

        ranked_lists = [
            vector_results,
            bm25_results,
        ]

        for results in ranked_lists:

            for rank, result in enumerate(
                results,
                start=1,
            ):
                result_id = result["id"]

                scores[result_id] += (
                    1 / (self.rrf_k + rank)
                )

                result_lookup[result_id] = result

        # --------------------------------------------------
        # Build candidate pool
        # --------------------------------------------------

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        candidates = []

        for result_id in ranked_ids:

            result = result_lookup[
                result_id
            ].copy()

            result["rrf_score"] = scores[
                result_id
            ]

            candidates.append(result)

        # --------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        return self.context_selector.select(
            candidates=reranked_results,
            top_k=top_k,
        )