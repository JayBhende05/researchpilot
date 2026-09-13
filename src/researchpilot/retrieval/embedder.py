from sentence_transformers import SentenceTransformer

from researchpilot.config.settings import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    #chunks --> vector
    #Normalization make the similarity caluclation easier (Magnitude = No, Direction = Yes)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    #query--> vector
    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()