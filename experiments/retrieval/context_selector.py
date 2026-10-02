import numpy as np


class ContextSelector:
    """
    Removes highly redundant chunks while preserving
    the original reranker ordering.

    The first chunk is always kept because it is the
    highest-ranked result from the reranker.
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.85,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(
        embedding_a,
        embedding_b,
    ) -> float:

        a = np.array(embedding_a)
        b = np.array(embedding_b)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )

    def select(
        self,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not candidates:
            return []

        selected = []
        selected_embeddings = []

        for candidate in candidates:

            if len(selected) >= top_k:
                break

            embedding = self.embedder.embed_query(
                candidate["document"]
            )

            is_redundant = False

            for selected_embedding in selected_embeddings:

                similarity = self.cosine_similarity(
                    embedding,
                    selected_embedding,
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):
                    is_redundant = True
                    break

            if is_redundant:
                continue

            selected.append(candidate)
            selected_embeddings.append(
                embedding
            )

        return selected