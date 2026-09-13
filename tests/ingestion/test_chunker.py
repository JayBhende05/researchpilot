from researchpilot.ingestion.chunker import chunk_documents



def test_data_chunker():
  documents = [
     {
            "text": "A" * 2000,
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        }
  ]


  chunks = chunk_documents(documents)
  assert len(chunks) > 1
  assert chunks[0]["metadata"]["source"] == "test.pdf"
  assert chunks[0]["metadata"]["page"] == 1

  
  