# from collections import defaultdict


# class RerankedRetriever:
#     """
#     Hybrid retrieval followed by cross-encoder reranking.

#     Pipeline:
#         Vector Search
#             +
#         BM25 Search
#             ↓
#         RRF Fusion
#             ↓
#         Candidate Pool
#             ↓
#         Cross-Encoder Reranker
#             ↓
#         Final Top-K
#     """

#     def __init__(
#         self,
#         vector_store,
#         embedder,
#         bm25_retriever,
#         reranker,
#         context_selector=None,
#         rrf_k: int = 60,
#     ):
#         self.vector_store = vector_store
#         self.embedder = embedder
#         self.bm25_retriever = bm25_retriever
#         self.reranker = reranker
#         self.context_selector = context_selector
#         self.rrf_k = rrf_k

#     def search(
#         self,
#         query: str,
#         top_k: int = 5,
#         candidate_k: int = 20,
#     ) -> list[dict]:

#         # --------------------------------------------------
#         # Vector retrieval
#         # --------------------------------------------------

#         query_embedding = self.embedder.embed_query(query)

#         vector_results = self.vector_store.search_ranked(
#             query_embedding=query_embedding,
#             top_k=candidate_k,
#         )

#         # --------------------------------------------------
#         # BM25 retrieval
#         # --------------------------------------------------

#         bm25_results = self.bm25_retriever.search(
#             query=query,
#             top_k=candidate_k,
#         )

#         # --------------------------------------------------
#         # RRF fusion
#         # --------------------------------------------------

#         scores = defaultdict(float)
#         result_lookup = {}

#         ranked_lists = [
#             vector_results,
#             bm25_results,
#         ]

#         for results in ranked_lists:

#             for rank, result in enumerate(
#                 results,
#                 start=1,
#             ):
#                 result_id = result["id"]

#                 scores[result_id] += (
#                     1 / (self.rrf_k + rank)
#                 )

#                 result_lookup[result_id] = result

#         # --------------------------------------------------
#         # Build candidate pool
#         # --------------------------------------------------

#         ranked_ids = sorted(
#             scores,
#             key=scores.get,
#             reverse=True,
#         )

#         candidates = []

#         for result_id in ranked_ids:

#             result = result_lookup[
#                 result_id
#             ].copy()

#             result["rrf_score"] = scores[
#                 result_id
#             ]

#             candidates.append(result)

#         # --------------------------------------------------
#         # Cross-encoder reranking
#         # --------------------------------------------------

#         reranked_results = self.reranker.rerank(
#             query=query,
#             candidates=candidates,
#             top_k=top_k,
#         )

#         if self.context_selector is not None:
#             return self.context_selector.select(
#                 candidates=reranked_results,
#                 top_k=top_k,
#             )

#         return reranked_results
        

# **************  PRINT EVERY STEP OUTPUT **************************

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
        context_selector=None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.context_selector = context_selector
        self.rrf_k = rrf_k

    def _print_result(
        self,
        rank,
        result,
        score_key=None,
    ):
        """
        Print a short, readable result preview.
        """

        metadata = result.get("metadata", {})

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        score = ""

        if score_key is not None:
            value = result.get(score_key)

            if value is not None:
                score = f" | {score_key}={value:.4f}"

        # Try common chunk/text field names
        text = (
            result.get("text")
            or result.get("content")
            or result.get("document")
            or ""
        )

        # Keep output short
        text = " ".join(
            str(text).split()
        )

        if len(text) > 180:
            text = text[:180] + "..."

        print(
            f"  #{rank} "
            f"source={source} "
            f"page={page}"
            f"{score}"
        )

        if text:
            print(
                f"      {text}"
            )

    def _print_results(
        self,
        title,
        results,
        score_key=None,
        limit=None,
    ):
        """
        Print a short list of retrieval results.
        """

        print("\n" + "-" * 60)
        print(title)
        print("-" * 60)

        if not results:
            print("  No results")
            return

        display_results = (
            results[:limit]
            if limit is not None
            else results
        )

        for rank, result in enumerate(
            display_results,
            start=1,
        ):
            self._print_result(
                rank=rank,
                result=result,
                score_key=score_key,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:

        # ==================================================
        # QUERY
        # ==================================================

        print("\n" + "=" * 60)
        print("RETRIEVAL PIPELINE")
        print("=" * 60)

        print(
            f"\nQuery:\n  {query}"
        )

        # ==================================================
        # Vector retrieval
        # ==================================================

        query_embedding = self.embedder.embed_query(
            query
        )

        vector_results = (
            self.vector_store.search_ranked(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        )

        self._print_results(
            title=(
                f"1. VECTOR RETRIEVAL "
                f"(Top-{candidate_k})"
            ),
            results=vector_results,
            limit=5,
        )

        print(
            f"\n  Vector candidates: "
            f"{len(vector_results)}"
        )

        # ==================================================
        # BM25 retrieval
        # ==================================================

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        self._print_results(
            title=(
                f"2. BM25 RETRIEVAL "
                f"(Top-{candidate_k})"
            ),
            results=bm25_results,
            limit=5,
        )

        print(
            f"\n  BM25 candidates: "
            f"{len(bm25_results)}"
        )

        # ==================================================
        # RRF fusion
        # ==================================================

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

        # ==================================================
        # Build candidate pool
        # ==================================================

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

        self._print_results(
            title=(
                f"3. RRF FUSION / "
                f"CANDIDATE POOL "
                f"(Top-{candidate_k})"
            ),
            results=candidates,
            score_key="rrf_score",
            limit=5,
        )

        print(
            f"\n  Unique candidates after RRF: "
            f"{len(candidates)}"
        )

        # ==================================================
        # Cross-encoder reranking
        # ==================================================

        reranked_results = (
            self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        )

        self._print_results(
            title=(
                f"4. CROSS-ENCODER RERANKER "
                f"(Top-{top_k})"
            ),
            results=reranked_results,
            limit=top_k,
        )

        # ==================================================
        # Context selector
        # ==================================================

        if self.context_selector is not None:

            selected_results = (
                self.context_selector.select(
                    candidates=reranked_results,
                    top_k=top_k,
                )
            )

            self._print_results(
                title=(
                    f"5. CONTEXT SELECTOR "
                    f"(Top-{top_k})"
                ),
                results=selected_results,
                limit=top_k,
            )

            return selected_results

        # ==================================================
        # Final results
        # ==================================================

        print("\n" + "-" * 60)
        print("FINAL RETRIEVAL")
        print("-" * 60)

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            metadata = result.get(
                "metadata",
                {},
            )

            print(
                f"  #{rank} "
                f"{metadata.get('source', 'unknown')} "
                f"| page={metadata.get('page', 'unknown')}"
            )

        print("=" * 60)

        return reranked_results
