from researchpilot.ingestion.embedder import Embedder


def test_embedding():
    embedder = Embedder()
    
    texts = [
        "LoRA reduces the number of trainable parameters.",
        "Low rank adaptation makes fine tuning more parameter efficient.",
        "The Eiffel Tower is located in Paris.",
    ]
    
    embeddings = embedder.embed_documents(texts)
    
    assert len(embeddings) > 0
    assert len(embeddings[0]) > 0
    



