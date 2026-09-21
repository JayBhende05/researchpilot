from collections import defaultdict


class ExpandedRerankedRetriever:
    """
    Retrieval pipeline:

        Original Query
              +
        Expanded Query
              ↓
        Vector + BM25
              ↓
            RRF
              ↓
        Candidate Pool
              ↓
        Cross-Encoder
              ↓
            Top-K
    """

    def __init__(
        self,
        vector_store,
        embedder,
        bm25_retriever,
        reranker,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.rrf_k = rrf_k

    def _retrieve_for_query(
        self,
        query: str,
        candidate_k: int,
    ):
        query_embedding = (
            self.embedder.embed_query(query)
        )

        vector_results = (
            self.vector_store.search_ranked(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        return [
            vector_results,
            bm25_results,
        ]

    def search(
        self,
        original_query: str,
        expanded_query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ):

        all_ranked_lists = []

        # --------------------------------------------------
        # Original query
        # --------------------------------------------------

        original_results = self._retrieve_for_query(
            query=original_query,
            candidate_k=candidate_k,
        )

        all_ranked_lists.extend(
            original_results
        )

        # --------------------------------------------------
        # Expanded query
        # --------------------------------------------------

        expanded_results = self._retrieve_for_query(
            query=expanded_query,
            candidate_k=candidate_k,
        )

        all_ranked_lists.extend(
            expanded_results
        )

        # --------------------------------------------------
        # RRF
        # --------------------------------------------------

        scores = defaultdict(float)
        result_lookup = {}

        for results in all_ranked_lists:

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
        # Candidate pool
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

            result["rrf_score"] = (
                scores[result_id]
            )

            candidates.append(result)

        # --------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------

        return self.reranker.rerank(
            query=original_query,
            candidates=candidates,
            top_k=top_k,
        )