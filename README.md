# ResearchPilot v0.3 — Grounded Answer Generation

**v0.3** builds on the improved retrieval pipeline from **v0.2** by adding end-to-end answer generation with Gemini and citation validation.

- **v0.2** focused on retrieving the right evidence.
- **v0.3** connects that evidence to a complete generation pipeline.

```text
Question
   ↓
Hybrid Retrieval + Reranking
   ↓
Top-5 Evidence Chunks
   ↓
Prompt Builder
   ↓
Gemini
   ↓
Structured Answer + Citation IDs
   ↓
Citation Validation
   ↓
RAGResponse
```

---

## What We Added

### 1. Generation Layer

Added a dedicated `generation/` module responsible for:

- Gemini integration
- Prompt construction
- Structured response schemas
- Converting retrieved evidence into generation context

The generation layer receives the **Top-5 chunks** produced by the v0.2 retrieval pipeline and asks Gemini to answer using that evidence.

---

### 2. End-to-End RAG Pipeline

Added:

```text
pipeline/rag.py
```

This connects the complete pipeline:

```text
Question
   ↓
Retrieval
   ↓
Top-5 Evidence
   ↓
Prompt Builder
   ↓
Gemini
   ↓
Structured Response
   ↓
Citation Validation
   ↓
RAGResponse
```

The goal is to make ResearchPilot a complete **retrieval-augmented generation (RAG) system** rather than a retrieval-only system.

---

### 3. Structured Gemini Output

Gemini is required to return a structured response containing both:

- The generated answer
- The IDs of the evidence chunks used to support it

Example:

```json
{
  "answer": "The model was trained using AdamW...",
  "citation_ids": ["chunk_1", "chunk_3"]
}
```

This separates the generated answer from its supporting evidence and makes citations **machine-readable**.

---

### 4. Citation Validation

Generated citation IDs are validated against the chunks that were actually retrieved.

Conceptually:

```text
Retrieved Chunks
      ↓
{chunk_1, chunk_2, chunk_3, chunk_4, chunk_5}
      ↓
             Gemini
               ↓
citation_ids: ["chunk_1", "chunk_3"]
               ↓
       Citation Validation
               ↓
       Validated Citations
```

Gemini cannot introduce arbitrary citation IDs that were not present in the retrieved context.

This provides an important grounding constraint:

> **The model can only cite evidence that the retrieval pipeline actually supplied.**

---

### 5. End-to-End Integration Test

Added an integration test covering the complete RAG flow.

The test verifies that:

1. A question enters the pipeline.
2. Retrieval produces evidence chunks.
3. The prompt builder constructs the generation context.
4. Gemini produces the expected structured response.
5. Citation IDs are validated against retrieved chunks.
6. The final `RAGResponse` is returned successfully.

This tests the system as an **integrated pipeline** rather than testing retrieval and generation independently.

---

# Architecture

The v0.2 retrieval architecture remains unchanged, with the generation and citation-validation stages added downstream.

```text
User Question
      │
      ▼
Vector Search ──────────────┐
Top-20                      │
                            ├──→ RRF Fusion
BM25 Search ────────────────┘     (k = 60)
Top-20
                                  │
                                  ▼
                           Candidate Pool
                                  │
                                  ▼
                         BGE Cross-Encoder
                            Reranking
                                  │
                                  ▼
                               Top-5
                          Evidence Chunks
                                  │
                                  ▼
                           Prompt Builder
                                  │
                                  ▼
                              Gemini
                                  │
                                  ▼
                    Structured Answer + IDs
                                  │
                                  ▼
                       Citation Validation
                                  │
                                  ▼
                             RAGResponse
```

---

# Pipeline Flow

The complete ResearchPilot v0.3 pipeline is:

```text
                         User Question
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
           Vector Search              BM25 Search
              Top-20                    Top-20
                 │                         │
                 └────────────┬────────────┘
                              ▼
                         RRF Fusion
                         (k = 60)
                              │
                              ▼
                       Candidate Pool
                              │
                              ▼
                  BGE Cross-Encoder
                     Reranking
                              │
                              ▼
                           Top-5
                              │
                              ▼
                       Prompt Builder
                              │
                              ▼
                           Gemini
                              │
                              ▼
                 Structured Answer + IDs
                              │
                              ▼
                    Citation Validation
                              │
                              ▼
                         RAGResponse
```

---

# Why This Change Matters

The v0.2 benchmark established that ResearchPilot could retrieve relevant evidence reliably:

| Metric | v0.1 | v0.2 |
|---|---:|---:|
| Document Top-1 | 100% | 100% |
| Evidence Top-1 | 55% | 62.5% |
| Evidence Top-3 | 70% | 90% |
| Evidence Top-5 | 80% | 92.5% |

The next problem is no longer only retrieval.

Once useful evidence has been retrieved, the system needs to:

- Synthesize that evidence into an answer.
- Preserve the connection between claims and retrieved evidence.
- Produce citations in a structured format.
- Prevent unsupported citation references.

**v0.3 addresses this first generation-stage problem.**

---

# v0.2 → v0.3

| Area | v0.2 | v0.3 |
|---|---|---|
| Retrieval | Hybrid Vector + BM25 | Unchanged |
| Fusion | RRF (`k = 60`) | Unchanged |
| Ranking | BGE Cross-Encoder | Unchanged |
| Evidence | Top-5 chunks | Top-5 chunks |
| Generation | — | Gemini |
| Output | Retrieved chunks | Structured answer + citation IDs |
| Citations | Retrieval evaluation | Citation validation |
| Integration | Retrieval pipeline | End-to-end RAG pipeline |
| Testing | Retrieval benchmark | End-to-end integration test |

---

# Engineering Lesson

v0.2 showed that **retrieval quality and ranking quality are separate problems**.

v0.3 extends that principle to generation:

> **A good RAG system needs both strong retrieval and controlled generation.**

Retrieving the correct evidence is not enough.

The generation layer must use that evidence in a way that remains:

- **Traceable**
- **Verifiable**
- **Grounded**

The current architecture therefore treats:

```text
Retrieval
    ↓
Generation
    ↓
Citation Validation
```

as separate stages with explicit interfaces between them.

---

# Current Limitations

The generation layer is now integrated, but answer-generation quality still needs its own evaluation.

Future work includes:

- Evaluating answer correctness against `expected_answer`.
- Measuring citation correctness and citation completeness.
- Detecting unsupported claims in generated answers.
- Evaluating whether citations actually support the claims they are attached to.
- Expanding the benchmark beyond 40 questions and 4 papers.
- Moving from page-level evidence labels toward chunk-level evidence evaluation.

---

# Next Evaluation Target

The next evaluation target is therefore not simply:

> **Did we retrieve the right evidence?**

but:

> **Did we generate the right answer, and can every important claim be traced back to retrieved evidence?**

This marks the transition from evaluating ResearchPilot as a **retrieval system** to evaluating it as a **grounded, end-to-end RAG system**.

---

## ResearchPilot Progression

```text
v0.1 — Basic / Naive RAG
        │
        ▼
   Vector Retrieval
        │
        ▼
     LLM Answer
        │
        │
        ▼
v0.2 — Better Retrieval
        │
        ├── Vector Search
        ├── BM25
        ├── RRF Fusion
        ├── BGE Reranking
        └── Top-5 Evidence
        │
        ▼
v0.3 — Grounded Answer Generation
        │
        ├── Retrieval
        ├── Prompt Builder
        ├── Gemini
        ├── Structured Output
        └── Citation Validation
        │
        ▼
   Grounded RAG Response
```