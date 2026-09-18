from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not candidates:
            return []

        pairs = [
            (query, candidate["document"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        ranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["reranker_score"] = float(score)
            ranked.append(result)

        ranked.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return ranked[:top_k]