
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 lexical retriever over the chunks stored in ChromaDB.

    The retriever is initialized with the complete chunk corpus.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.tokenized_corpus = [
            self._tokenize(chunk["document"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple tokenizer for technical paper text.
        """

        text = text.lower()

        # Keep words, numbers, and hyphenated terms.
        tokens = re.findall(
            r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
            text,
        )

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k chunks using BM25.
        """

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()

            result["bm25_score"] = float(
                scores[index]
            )

            results.append(result)

        return results

