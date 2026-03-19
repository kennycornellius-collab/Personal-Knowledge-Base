import streamlit as st
import tempfile
import os
from ingestion import HybridIndexer
from openai import OpenAI
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
    
    st.divider()
    st.header("4. Synthesized Answer (Local LLM)")
    
    col_llm1, col_llm2, col_llm3 = st.columns(3)
    with col_llm1:
        llm_base_url = st.text_input("Local Base URL", value="http://localhost:8000/v1") 
    with col_llm2:
        llm_model_name = st.text_input("Model Name", value="local-model")
    with col_llm3:
        llm_api_key = st.text_input("API Key", type="password", value="")

    if st.button("Generate Answer from Hybrid Context"):
        context_texts = []
        for i, res in enumerate(results_hybrid, 1):
            source = res.get("payload", {}).get("source", "Unknown")
            text = res.get("payload", {}).get("text", "")
            context_texts.append(f"--- Document [{i}] (Source: {source}) ---\n{text}")
        
        context_block = "\n\n".join(context_texts)
        
        system_prompt = """You are a helpful technical assistant. Answer the user's question based ONLY on the provided context. 
        If the context does not contain the answer, explicitly state 'I cannot answer this based on the provided documents.'
        Always cite the Document number when making a claim."""
        
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"
        
        client = OpenAI(base_url=llm_base_url, api_key=llm_api_key if llm_api_key else "dummy-key")
        
        def token_generator(stream):
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        try:
            st.markdown("### Answer")
            response = client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                stream=True
            )
            
            st.write_stream(token_generator(response))
                
        except Exception as e:
            st.error(f"Failed to connect to Local LLM at {llm_base_url}")
            st.error(f"Detailed error: {e}")