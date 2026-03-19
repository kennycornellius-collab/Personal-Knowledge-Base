import streamlit as st
import tempfile
import os
from ingestion import HybridIndexer

st.set_page_config(page_title="RAG Fusion Dashboard", layout="wide")
st.title("Hybrid RAG: Retrieval Quality Dashboard")

if "indexer" not in st.session_state:
    st.session_state.indexer = HybridIndexer()
    st.session_state.is_ready = False

indexer: HybridIndexer = st.session_state.indexer

with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs or Markdown", type=["pdf", "md", "txt"], accept_multiple_files=True)
    
    chunk_size = st.number_input("Chunk Size (chars)", value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", value=50, step=10)
    
    if st.button("Ingest Documents") and uploaded_files:
        with st.spinner("Chunking and Dual-Indexing..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                
                indexer.ingest_document(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source_name=uploaded_file.name)
                os.unlink(tmp_path)
                
        st.session_state.is_ready = True
        st.success(f"Successfully ingested {len(uploaded_files)} files!")
        
    if st.session_state.is_ready:
        st.divider()
        st.metric("Chunks in Vector Store (Qdrant)", indexer.qdrant.count(indexer.collection_name).count)
        st.metric("Chunks in Keyword Store (BM25)", len(indexer.bm25_id_map))

if not st.session_state.is_ready:
    st.info("Please upload and ingest some documents in the sidebar to get started.")
    st.stop()

st.header("2. Search & Evaluate")
query = st.text_input("Enter your query:", placeholder="e.g., What is the difference between dense and sparse retrieval?")

col_tune1, col_tune2 = st.columns([2, 1])
with col_tune1:
    alpha = st.slider(
        "Fusion Weight (Alpha) — 0.0 is Pure BM25, 1.0 is Pure Vector", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
with col_tune2:
    top_k = st.number_input("Results to display (Top-K)", min_value=1, max_value=10, value=3)


def render_result_card(result, index):
    payload = result.get("payload", {})
    text = payload.get("text", "Text missing")
    source = payload.get("source", "Unknown")

    dense_rank = result.get('dense_rank') or "—"
    sparse_rank = result.get('sparse_rank') or "—"
    
    st.markdown(f"**#{index} | {source}**")
    st.caption(f"Score: `{result['fused_score']:.4f}` | Dense Rank: `{dense_rank}` | Sparse Rank: `{sparse_rank}`")
    st.info(text)
    st.divider()

if query and query.strip():
    st.header("3. Retrieval Comparison")
    col_dense, col_hybrid, col_sparse = st.columns(3)
    
    with col_dense:
        st.subheader("Dense Only (alpha = 1.0)")
        results_dense = indexer.hybrid_search(query, top_k=top_k, alpha=1.0)
        for i, res in enumerate(results_dense, 1):
            render_result_card(res, i)
            
    with col_hybrid:
        st.subheader(f"Hybrid RRF (alpha = {alpha})")
        results_hybrid = indexer.hybrid_search(query, top_k=top_k, alpha=alpha)
        for i, res in enumerate(results_hybrid, 1):
            render_result_card(res, i)
            
    with col_sparse:
        st.subheader("Sparse Only (alpha = 0.0)")
        results_sparse = indexer.hybrid_search(query, top_k=top_k, alpha=0.0)
        for i, res in enumerate(results_sparse, 1):
            render_result_card(res, i)