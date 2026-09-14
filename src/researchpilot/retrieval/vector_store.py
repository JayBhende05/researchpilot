import chromadb

from researchpilot.config.settings import COLLECTION_NAME



class VectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        reset: bool = False,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        if reset:
            try:
                self.client.delete_collection(
                    name=COLLECTION_NAME
                )
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:

        ids = []
        documents = []
        metadatas = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            metadata = chunk["metadata"]

            chunk_id = (
                f"{metadata['source']}"
                f"_p{metadata['page']}"
                f"_c{metadata['chunk']}"
            )

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )