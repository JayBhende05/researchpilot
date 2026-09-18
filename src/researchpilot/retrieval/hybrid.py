
from collections import defaultdict


class HybridRetriever:
    """
    Combines vector retrieval and BM25 retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store,
        embedder,
        bm25_retriever,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict]:
        """
        Retrieve candidates using both vector search
        and BM25, then combine rankings with RRF.
        """

        # --------------------------------------------------
        # Vector retrieval
        # --------------------------------------------------

        query_embedding = (
            self.embedder.embed_query(query)
        )

        vector_results = (
            self.vector_store.search_ranked(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        )

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        # --------------------------------------------------
        # RRF
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

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        final_results = []

        for result_id in ranked_ids[:top_k]:

            result = result_lookup[
                result_id
            ].copy()

            result["rrf_score"] = (
                scores[result_id]
            )

            final_results.append(result)

        return final_results

