# Hybrid RAG: Retrieval Quality Dashboard

Most RAG (Retrieval-Augmented Generation) projects stop at dense vector search. Pure semantic search has a well-documented failure mode — it struggles with exact keyword matches, proper nouns, and version numbers.

This project is a fully local Hybrid RAG pipeline that solves this by combining **Dense Retrieval (vector)** and **Sparse Retrieval (keyword)**. The core differentiator is an interactive visual dashboard that exposes the internals of the retrieval process, letting you compare strategies side-by-side and tune the fusion weight before passing context to a local LLM.

---

## Architecture

| Component | Technology |
|---|---|
| UI & Dashboard | `Streamlit` |
| Dense Vector Store | `Qdrant` (persistent, local disk) |
| Dense Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Sparse Keyword Index | `rank_bm25` (persisted as JSON) |
| Document Parsing | `pypdf`, LangChain text splitters |
| Generation | Custom local inference server (`FastAPI` + `llama-cpp-python`, CUDA) |

---

## How It Works

**Dual ingestion.** Documents (PDF or Markdown) are chunked using recursive character splitting. The exact same chunks — linked by a universal UUID — are simultaneously embedded into Qdrant and tokenized into BM25, ensuring both indexes operate on identical source material. This is a hard architectural requirement: Reciprocal Rank Fusion works by merging ranked lists of the same items by UUID. Different chunk sets for each index would make fusion meaningless.

**Tunable Reciprocal Rank Fusion.** Queries run against both indexes in parallel. Results are merged using a weighted RRF formula:

```
Score = (α × 1/(k + rank_dense)) + ((1-α) × 1/(k + rank_sparse))
```

A UI slider lets you adjust the alpha weight from `0.0` (pure BM25) to `1.0` (pure vector) and observe the effect in real time.

**Retrieval Quality Dashboard.** The UI renders three columns — Dense Only, Hybrid, and Sparse Only — annotating each chunk with its raw score, dense rank, and sparse rank. This makes it possible to visually track how individual chunks move through the fused list as you adjust the weighting.

**Fully local generation.** The top-k fused chunks are injected into a strict system prompt and sent to a local authenticated LLM inference server. No data leaves the machine.

**Persistent storage.** Qdrant writes to a local RocksDB-backed folder. The BM25 index state (`id_map`, tokenized corpus, and ingested source registry) is serialized to JSON alongside it. Both are restored on startup with a sync validation check. Duplicate ingestion of the same file is automatically skipped.

**Automated evaluation.** A synthetic Q&A harness uses the local LLM to generate conceptual questions from random corpus chunks, then measures Hit Rate and MRR across all three retrieval strategies programmatically.

---

## Benchmark Results

Evaluated on 10 synthetic questions generated from a technical Markdown document, Top-K = 3.

| Method | Hit Rate | MRR |
|---|---|---|
| Dense Only (α=1.0) | 90.00% | 0.900 |
| Hybrid RRF (α=0.5) | 80.00% | 0.683 |
| Sparse Only (α=0.0) | 70.00% | 0.600 |

> **Note on these numbers.** The synthetic Q&A generator uses a concept-focused prompt specifically designed to reduce BM25 bias — questions are generated using different vocabulary than the source text. Despite this, dense search led on this particular corpus. Results will vary by document type: BM25 typically gains ground on technical corpora with precise terminology (version numbers, API names, exact error strings).

---

## Engineering Observations

**Instruction following on citations was weak.** Despite strict system prompts requiring explicit document citations (e.g., `[1]`), smaller 7B/8B class models frequently dropped citation formatting mid-generation. This reflects a known trade-off between local models and frontier models like GPT-4, where the latter maintain tighter attention to formatting constraints over longer outputs.

**Chunking strategy is a shared constraint.** An early design considered different chunk sizes for dense vs. sparse indexing (smaller chunks for BM25, larger for embeddings). This is architecturally invalid — RRF requires both indexes to operate on the exact same UUID space. The correct approach for recovering the "small chunks retrieve better, large chunks generate better" insight is Parent-Child Retrieval (search on child chunks, return parent context to the LLM).

**BM25 has no incremental update API.** Every document ingestion rebuilds the BM25 index from the full corpus. This is by design in `rank_bm25` — the index is a computed statistic over the entire corpus, not an appendable structure. For small-to-medium corpora this is negligible. For very large corpora, migrating to an Elasticsearch-backed sparse index would be the correct path.

---

## Project Structure

```
Personal-Knowledge-Base/
├── ingestion.py        # HybridIndexer class — dual ingestion, retrieval, fusion
├── evaluate.py         # Synthetic Q&A generation, Hit Rate & MRR evaluation
├── app.py              # Streamlit dashboard — search, comparison, generation, benchmark
├── db_storage/         # Persistent Qdrant store + BM25 JSON state (gitignored)
├── .env                # LLM server config (not committed)
├── requirements.txt    # Pinned dependencies
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL_NAME=your-model-name
LLM_API_KEY=your-api-key
```

### Running the Dashboard

1. Ensure your local LLM inference server is running and accessible.

2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a document in the sidebar, adjust your chunking parameters, and click **Ingest**. Previously ingested documents are skipped automatically on restart.

4. In the **Interactive Search & QA** tab: query the corpus and tune the alpha slider to compare retrieval strategies side-by-side.

5. In the **Automated Evaluation** tab: run the quantitative benchmark to generate synthetic questions and compute Hit Rate and MRR across all three retrieval methods.

---

## Roadmap

- [x] Dual ingestion pipeline (same UUID across Qdrant and BM25)
- [x] Weighted Reciprocal Rank Fusion with tunable alpha slider
- [x] Side-by-side retrieval comparison dashboard (Dense / Hybrid / Sparse)
- [x] Grounded LLM generation with source citations
- [x] Quantitative evaluation harness (Hit Rate & MRR)
- [x] Persistent storage (Qdrant disk + BM25 JSON state)
- [x] Duplicate ingestion guard