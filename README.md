# Hybrid RAG: Retrieval Quality Dashboard

Most RAG (Retrieval-Augmented Generation) projects stop at dense vector search. Pure semantic search has a well-documented failure mode — it struggles with exact keyword matches, proper nouns, and version numbers.

This project is a fully local Hybrid RAG pipeline that solves this by combining **Dense Retrieval (vector)** and **Sparse Retrieval (keyword)**. The core differentiator is an interactive visual dashboard that exposes the internals of the retrieval process, letting you compare strategies side-by-side and tune the fusion weight before passing context to a local LLM.

---

## Architecture

| Component | Technology |
|---|---|
| UI & Dashboard | `Streamlit` |
| Dense Vector Store | `Qdrant` (in-memory) |
| Dense Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Sparse Keyword Index | `rank_bm25` (in-memory) |
| Document Parsing | `pypdf`, LangChain text splitters |
| Generation | Custom local inference server (`FastAPI` + `llama-cpp-python`, CUDA) |

---

## How It Works

**Dual ingestion.** Documents (PDF or Markdown) are chunked using recursive character splitting. The exact same chunks — linked by a universal UUID — are simultaneously embedded into Qdrant and tokenized into BM25, ensuring both indexes operate on identical source material.

**Tunable Reciprocal Rank Fusion.** Queries run against both indexes in parallel. Results are merged using a mathematically weighted RRF algorithm. A UI slider lets you adjust the alpha weight from `0.0` (pure BM25) to `1.0` (pure vector) and observe the effect in real time.

**Retrieval Quality Dashboard.** The UI renders three columns — Dense Only, Hybrid, and Sparse Only — annotating each chunk with its raw score, dense rank, and sparse rank. This makes it possible to visually track how individual chunks move up or down the fused list as you adjust the weighting.

**Fully local generation.** The top-k fused chunks are injected into a strict system prompt and sent to a local authenticated LLM inference server. No data leaves the machine.

---

## Engineering Observations: Local LLM Performance

During testing with 7B/8B class models hosted via `llama-cpp-python`, two patterns emerged in the generation phase.

**Synthesis quality was strong.** Smaller local models performed well at extracting technical details from retrieved chunks and synthesizing them into accurate, fluid summaries.

**Instruction following on citations was weak.** Despite strict system prompts requiring explicit document citations (e.g., appending `[1]`), smaller models frequently dropped the citation formatting mid-generation. This reflects a known trade-off between local models and frontier models like GPT-4, where the latter maintain tighter attention to formatting constraints over longer outputs.

---

## Setup & Usage

### Requirements

```bash
pip install streamlit qdrant-client rank_bm25 sentence-transformers langchain-text-splitters pypdf openai
```

### Running the Dashboard

1. Ensure your local LLM inference server is running and accessible (default config points to `http://localhost:8000/v1`).

2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a document in the sidebar, adjust your chunking parameters, and click **Ingest**.

4. Query the corpus and tune the alpha slider to evaluate retrieval quality across strategies.

---

## Roadmap

**Quantitative Evaluation Harness:** Build an automated script to generate synthetic Q&A pairs from the corpus and programmatically calculate Hit Rate and Mean Reciprocal Rank (MRR) — providing hard mathematical proof of the hybrid approach's superiority over pure dense search.

**Persistent Storage:** Migrate Qdrant and the UUID mapping from in-memory to disk for handling larger document corpora across sessions.
