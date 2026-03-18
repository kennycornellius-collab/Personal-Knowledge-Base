import uuid
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

class HybridIndexer:
    def __init__(self, collection_name: str = "local_knowledge_base"):
        print("Initializing Local Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        print("Initializing Qdrant (In-Memory)")
        self.qdrant = QdrantClient(location=":memory:")
        self.collection_name = collection_name

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        self.bm25_id_map: List[str] = []       
        self.bm25_tokenized: List[List[str]] = [] 
        self.bm25_index = None

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

    def ingest_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        print(f"\n--- Ingesting: {Path(file_path).name} ---")
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
                "source": Path(file_path).name,
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
        
        print("Ingestion complete!")

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
    
    print("\n--- Multi-Doc Sanity Check ---")
    print(f"Total Chunks in Qdrant: {qdrant_count}")
    print(f"Total Chunks in BM25: {bm25_count}")
    print("Do they match? ", "YES!" if qdrant_count == bm25_count else "NO!")