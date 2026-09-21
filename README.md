# ResearchPilot v0.1 — Baseline Retrieval

v0.1 is the first working version of ResearchPilot's retrieval pipeline: a simple dense-vector RAG baseline over technical research papers. It is the reference point that every v0.2 experiment is measured against.

## Architecture

```
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Top-K
```

## Configuration

| Setting    | Value                     |
|------------|---------------------------|
| Chunking   | 800 characters per chunk  |
| Overlap    | 120 characters            |
| Retrieval  | Dense vector similarity   |

## Evaluation Benchmark

v0.1 was the version used to establish the evaluation foundation (see the v0.2 README, milestone M1):

- **40** research questions
- **4** technical papers
- `expected_source` — the correct document for each question
- `gold_evidence` — page-level evidence labels
- `expected_answer` — reserved for future answer-generation evaluation

## Baseline Results

| Level              | Top-1 | Top-3 | Top-5 |
|--------------------|-------|-------|-------|
| Document retrieval | 100%  | 100%  | 100%  |
| Evidence retrieval | 33%   | 41%   | 66%   |

Gold evidence **not** in Top-5: **20%**

## Key Finding

Document retrieval was already strong: the system reliably found the correct paper. The main weakness was **evidence ranking** — relevant evidence was often found but ranked below the top result, and 20% of the time did not appear in the Top-5 at all. This finding drove the goals of v0.2.

## Known Limitations

- Dense-only retrieval misses exact technical terminology.
- No re-ranking stage; ordering relies purely on embedding similarity.
- Fixed-size character chunks can split content mid-thought (though paragraph-aware chunking did not beat this in v0.2 testing).
- Benchmark uses coarse page-level evidence labels.

## What's Next

See [`README.md`](README.md) for the experiments and the resulting hybrid + reranking pipeline.