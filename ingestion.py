import uuid
import re
import json 
import os   
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

class HybridIndexer:
    def __init__(self, collection_name: str = "local_knowledge_base", persist_dir: str = "./db_storage"):
        print("Initializing Local Embedding Model (all-MiniLM-L6-v2)")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.bm25_state_file = Path(self.persist_dir) / "bm25_state.json"
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        print(f"Initializing Qdrant (Persistent at {self.persist_dir})...")

        self.qdrant = QdrantClient(path=self.persist_dir)

        if not self.qdrant.collection_exists(self.collection_name):
            print(f"Creating new Qdrant collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Loaded existing Qdrant collection: {self.collection_name}")

        self._load_bm25_state()

    def _load_bm25_state(self):
        if self.bm25_state_file.exists():
            print("Loading existing BM25 keyword index from disk...")
            with open(self.bm25_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.bm25_id_map = state.get("id_map", [])
                self.bm25_tokenized = state.get("tokenized", [])
                self.ingested_sources = set(state.get("ingested_sources", []))
                
            if self.bm25_tokenized:
                self.bm25_index = BM25Okapi(self.bm25_tokenized)
            else:
                self.bm25_index = None
        else:
            print("No existing BM25 state found. Starting fresh.")
            self.bm25_id_map: List[str] = []       
            self.bm25_tokenized: List[List[str]] = [] 
            self.bm25_index = None
            self.ingested_sources = set()

        qdrant_count = self.qdrant.count(self.collection_name).count
        if qdrant_count != len(self.bm25_id_map):
            print(f"WARNING: Qdrant ({qdrant_count}) and BM25 ({len(self.bm25_id_map)}) are out of sync. Consider clearing {self.persist_dir} and re-ingesting.")

    def _save_bm25_state(self):
        with open(self.bm25_state_file, 'w', encoding='utf-8') as f:
            json.dump({
                "id_map": self.bm25_id_map,
                "tokenized": self.bm25_tokenized,
                "ingested_sources": list(self.ingested_sources) 
            }, f)

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif path.suffix.lower() in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Unsupported file type. Please provide a .pdf, .md, or .txt file.")

    def tokenize_for_bm25(self, text: str) -> List[str]:
        return [word for word in re.split(r'\W+', text.lower()) if word]

    def ingest_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, source_name: str = None):
        display_name = source_name or Path(file_path).name
        
        if display_name in self.ingested_sources:
            print(f"Skipping '{display_name}' — already ingested.")
            return
            
        print(f"\n--- Ingesting: {display_name} ---")
        
        raw_text = self.extract_text(file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        print(f"Batch encoding {len(chunks)} chunks")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
        points = []
        print("Preparing payloads and updating indices")
        for i, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            metadata = {
                "source": display_name, 
                "chunk_index": i,
                "text": chunk_text
            }
            points.append(
                PointStruct(
                    id=chunk_id, 
                    vector=embeddings[i].tolist(), 
                    payload=metadata
                )
            )
            self.bm25_tokenized.append(self.tokenize_for_bm25(chunk_text))
            self.bm25_id_map.append(chunk_id)

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        self.bm25_index = BM25Okapi(self.bm25_tokenized)
        self.ingested_sources.add(display_name)
        self._save_bm25_state()

    def search_dense(self, query: str, limit: int = 50) -> List[dict]:
        query_vector = self.embedding_model.encode(query).tolist()
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector, 
            limit=limit
        )
        return [{"chunk_id": res.id, "score": res.score, "payload": res.payload} for res in response.points]

    def search_sparse(self, query: str, limit: int = 50) -> List[dict]:
        if not self.bm25_index:
            return []
            
        tokenized_query = self.tokenize_for_bm25(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: 
                results.append({
                    "chunk_id": self.bm25_id_map[idx], 
                    "score": scores[idx],
                    "payload": None 
                })
        return results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5, rrf_k: int = 60) -> List[dict]:
        
        pool_size = max(top_k * 3, 50) 

        dense_results = self.search_dense(query, limit=pool_size) if alpha > 0.0 else []
        sparse_results = self.search_sparse(query, limit=pool_size) if alpha < 1.0 else []
        
        dense_rank_map = {res["chunk_id"]: rank for rank, res in enumerate(dense_results, 1)}
        sparse_rank_map = {res["chunk_id"]: rank for rank, res in enumerate(sparse_results, 1)}
        
        payload_map = {res["chunk_id"]: res["payload"] for res in dense_results if res["payload"]}
        
        all_uuids = set(dense_rank_map.keys()).union(set(sparse_rank_map.keys()))

        missing_ids = [uid for uid in all_uuids if uid not in payload_map]
        if missing_ids:
            fetched_points = self.qdrant.retrieve(
                collection_name=self.collection_name, 
                ids=missing_ids
            )
            for point in fetched_points:
                payload_map[point.id] = point.payload

        fused_results = []
        for chunk_id in all_uuids:
            dense_rank = dense_rank_map.get(chunk_id, None)
            sparse_rank = sparse_rank_map.get(chunk_id, None)
            
            dense_rrf = 1.0 / (rrf_k + dense_rank) if dense_rank else 0.0
            sparse_rrf = 1.0 / (rrf_k + sparse_rank) if sparse_rank else 0.0
            
            final_score = (alpha * dense_rrf) + ((1.0 - alpha) * sparse_rrf)
            
            fused_results.append({
                "chunk_id": chunk_id,
                "fused_score": final_score,
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "payload": payload_map.get(chunk_id, {}) 
            })
            
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused_results[:top_k]

if __name__ == "__main__":
    pipeline = HybridIndexer()

    doc1 = "doc1_rag.md"
    doc2 = "doc2_llm.md"
    
    with open(doc1, "w") as f:
        f.write("RAG combines search with LLMs. Dense retrieval uses embeddings. Sparse uses BM25.")
    with open(doc2, "w") as f:
        f.write("Local LLMs like Llama-3 or Mistral can run on consumer hardware. They are great for privacy.")

    pipeline.ingest_document(doc1, chunk_size=50, chunk_overlap=10)
    pipeline.ingest_document(doc2, chunk_size=50, chunk_overlap=10)
    
    qdrant_count = pipeline.qdrant.count(pipeline.collection_name).count
    bm25_count = len(pipeline.bm25_id_map)
    
    print(f"\n--- Sanity Check ---")
    print(f"Qdrant: {qdrant_count} | BM25: {bm25_count} | Match: {'YES' if qdrant_count == bm25_count else 'NO'}")
    
    test_query = "What is BM25?"
    
    print("\n--- Pure BM25 (Alpha = 0.0) ---")
    res_sparse = pipeline.hybrid_search(test_query, top_k=2, alpha=0.0)
    for r in res_sparse:
        print(f"Score: {r['fused_score']:.4f} | Text: {r['payload']['text']}")

    print("\n--- Pure Vector (Alpha = 1.0) ---")
    res_dense = pipeline.hybrid_search(test_query, top_k=2, alpha=1.0)
    for r in res_dense:
        print(f"Score: {r['fused_score']:.4f} | Text: {r['payload']['text']}")

    print("\n--- Hybrid RRF (Alpha = 0.5) ---")
    res_hybrid = pipeline.hybrid_search(test_query, top_k=2, alpha=0.5)
    for r in res_hybrid:
        print(f"Score: {r['fused_score']:.4f} | Dense Rank: {r['dense_rank']} | Sparse Rank: {r['sparse_rank']} | Text: {r['payload']['text']}")