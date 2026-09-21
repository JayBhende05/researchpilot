# ResearchPilot v0.2 — Better Retrieval

v0.2 improves **evidence retrieval** over the v0.1 baseline while preserving 100% document retrieval. It replaces plain vector search with a **hybrid (Vector + BM25) retriever, RRF fusion, and cross-encoder reranking**.

## Highlights

- Evidence Top-1: **55% → 62.5%**
- Evidence Top-3: **70% → 90%**
- Evidence Top-5: **80% → 92.5%**
- Gold evidence missing from Top-5: **20% → 7.5%**
- Document retrieval unchanged at **100%** (Top-1/3/5)

## Architecture

```
                    User Question
                         │
                         ▼
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Vector Search           BM25 Search
          Top-20                  Top-20
              │                     │
              └──────────┬──────────┘
                         ▼
                    RRF Fusion
                    (RRF k = 60)
                         │
                         ▼
                  Candidate Pool
                         │
                         ▼
             BGE Cross-Encoder Reranking
             (BAAI/bge-reranker-base)
                         │
                         ▼
                      Top-5
```

Compared with v0.1:

```
v0.1:  Question → Embedding → Vector Search → Top-K

v0.2:  Question → Vector Search ──┐
                                  ├→ RRF → Candidate Pool → Cross-Encoder → Top-5 Evidence
                       BM25 ──────┘
```

RRF score: `Σ 1 / (60 + rank)`

## Final Metrics

Questions evaluated: **40** (4 technical papers)

| Level              | Top-1 | Top-3 | Top-5 |
|--------------------|-------|-------|-------|
| Document retrieval | 100%  | 100%  | 100%  |
| Evidence retrieval | 62.5% | 90%   | 92.5% |

Gold evidence not in Top-5: **3/40 = 7.5%**

## Methodology

**M1 — Evaluation foundation.** Moved from subjective judgment ("retrieval looks good") to a measurable benchmark: 40 questions, 4 papers, `expected_source` for document retrieval, page-level `gold_evidence` for evidence retrieval, and `expected_answer` reserved for future answer-generation evaluation. Metrics: Top-1/3/5 for both document and evidence retrieval.

**M2 — Failure analysis.** The baseline found the correct document but often ranked the relevant evidence below the top result. Document retrieval was strong; evidence retrieval was weak. Objective: *improve evidence retrieval without sacrificing document retrieval.*

Each experiment below tested one hypothesis at a time and was kept only if it produced a measurable improvement.

## Experiment Log

All numbers are Evidence Top-1 / Top-3 / Top-5 (%). Document retrieval was 100% throughout except where noted.

| ID | Experiment | Top-1 | Top-3 | Top-5 | Decision |
|----|------------|-------|-------|-------|----------|
| —  | v0.1 Baseline | 55 | 70 | 80 | Baseline |
| E1 | Structure-aware (paragraph) chunking | 50 | 70 | 80 | ❌ Reject |
| E2 | Naive query transformation | 47.5 | 80 | 85 | ❌ Reject |
| E3 | Multi-query retrieval + RRF | 52.5 | 80 | 85 | ❌ Reject |
| E4 | Hybrid Vector + BM25 (RRF) | 60 | 85 | 92.5 | ✅ Keep |
| E5 | Cross-encoder reranking | 62.5 | 90 | 92.5 | ✅ Keep |
| E6 | Candidate pool expansion (20 → 50) | 57.5 | 87.5 | 90 | ❌ Reject |
| E7 | Query expansion + reranking | 62.5 | 90 | 92.5 | ❌ Reject |
| E8 | Context selection (redundancy filtering) | — | — | — | ❌ Reject |

### E1 — Structure-Aware Chunking ❌
**Hypothesis:** Paragraph-aware chunks preserve semantic boundaries better than fixed 800/120 character chunks.
**Result:** Top-1 dropped 55% → 50%; Top-3 and Top-5 unchanged.
**Lesson:** More semantically structured chunks do not automatically retrieve better. Fixed-size chunking remained the better baseline for this corpus.

### E2 — Naive Query Transformation ❌
**Hypothesis:** Stripping stop words and question structure (e.g. "What optimizer was used to train the model?" → "optimizer train model") leaves the important keywords and improves retrieval.
**Result:** Top-3/Top-5 improved (80% / 85%), but evidence Top-1 fell to 47.5% and document Top-1 fell to 95%.
**Lesson:** Aggressive query transformation can improve recall while damaging ranking precision.

### E3 — Multi-Query Retrieval + RRF ❌
**Hypothesis:** Different formulations (original, keyword-oriented, focused) retrieve different relevant chunks; fusing them with RRF improves recall.
**Result:** Document retrieval preserved; Top-3/Top-5 improved (80% / 85%) but Top-1 fell to 52.5%, below baseline.
**Lesson:** Multi-query retrieval can increase recall without improving the ranking of the most relevant evidence.

### E4 — Hybrid Vector + BM25 ✅
**Hypothesis:** Technical papers contain both semantic concepts and exact terminology, so combining dense and lexical retrieval captures both.
**Configuration:** Vector candidates 10, BM25 candidates 10, RRF k = 60, final results 5.
**Result:** Top-1 +5 pp, Top-3 +15 pp, Top-5 +12.5 pp over v0.1; not-in-Top-5 fell from 20% to 7.5%. Document retrieval stayed at 100%.
**Lesson:** Combining semantic and lexical retrieval is highly effective for this technical-paper corpus.

### E5 — Cross-Encoder Reranking ✅
**Hypothesis:** The hybrid retriever gives a good candidate pool, but a cross-encoder (`BAAI/bge-reranker-base`) scores query–passage relevance better than independent embedding similarity.
**Pipeline:** Vector Top-20 + BM25 Top-20 → RRF → candidate pool → reranker → Top-5.
**Result vs. E4:** Top-1 +2.5 pp, Top-3 +5 pp, Top-5 unchanged.
**Lesson:** Retrieval and ranking are separate problems. Hybrid retrieval improves candidate recall; reranking improves candidate ordering.

### E6 — Candidate Pool Expansion ❌
**Hypothesis:** Growing the pool from 20 to 50 per retriever recovers evidence missing from the smaller pool.
**Result vs. E5:** Top-1 62.5% → 57.5%, Top-3 90% → 87.5%, Top-5 92.5% → 90%.
**Lesson:** Larger candidate pools add distractors and make ranking harder.

### E7 — Query Expansion + Reranking ❌
**Hypothesis:** Adding a keyword-focused query alongside the (preserved) original query improves evidence retrieval. Unlike E2, the original query was not replaced.
**Result:** Identical to E5 (62.5 / 90 / 92.5).
**Lesson:** Extra query formulations add nothing once hybrid retrieval and reranking are in place.

### E8 — Context Selection ❌
**Hypothesis:** Removing highly redundant chunks from the reranked Top-5 gives a cleaner evidence set.
**Result:** No measurable improvement over E5. The failure analysis also exposed a benchmark limitation: a retrieved chunk can contain the relevant evidence even when the gold page label points to an adjacent page.
**Lesson:** Don't add a post-processing stage unless it produces measurable improvement. Evaluation should eventually move from page-level labels to chunk-level evidence.

## Core Engineering Lesson

> Don't optimize RAG by continuously adding techniques. Establish a benchmark, identify the failure mode, test one hypothesis at a time, and keep only changes that produce measurable improvement.

## Known Limitations & Future Work

- **Page-level gold labels are coarse.** Evidence found on an adjacent page is scored as a miss; move toward chunk/evidence-level evaluation.
- **Top-1 evidence is still 62.5%.** Ranking the single best chunk remains the main opportunity.
- **3/40 questions** still have no gold evidence in the Top-5.
- **Answer generation is not yet evaluated.** `expected_answer` is reserved in the benchmark for this.
- The benchmark is small (40 questions, 4 papers); results should be validated on a larger corpus.

## Changes from v0.1

| Area | v0.1 | v0.2 |
|------|------|------|
| Retrieval | Dense vector only | Vector + BM25 |
| Fusion | — | RRF (k = 60) |
| Ranking | Embedding similarity | `BAAI/bge-reranker-base` cross-encoder |
| Candidates | Top-K | Top-20 per retriever → Top-5 final |
| Evaluation | Subjective | 40-question benchmark with document and evidence metrics |