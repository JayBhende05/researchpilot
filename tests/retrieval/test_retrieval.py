from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore

query = "What is Fine Tuning?"

embedder = Embedder()
vector_store = VectorStore()


query_embedding = embedder.embed_query(query)

results = vector_store.search(
    query_embedding=query_embedding,
    top_k=5,
)

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print("\n" + "=" * 80)
    print(f"Result #{i + 1}")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(f"Chunk: {metadata['chunk']}")
    print("-" * 80)
    print(document[:1000])