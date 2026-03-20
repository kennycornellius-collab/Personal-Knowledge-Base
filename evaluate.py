import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from ingestion import HybridIndexer

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-model")
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not LLM_API_KEY:
    raise ValueError("CRITICAL: LLM_API_KEY not found. Please set it in your .env file.")

NUM_EVAL_QUESTIONS = 10 
TOP_K = 3

def generate_synthetic_qa(indexer: HybridIndexer, num_questions: int) -> list:
    print(f"\n--- Generating {num_questions} Synthetic Q&A Pairs ---")
    
    
    all_uuids = indexer.bm25_id_map
    
    
    sample_ids = random.sample(all_uuids, min(num_questions, len(all_uuids)))
    
    points = indexer.qdrant.retrieve(
        collection_name=indexer.collection_name, 
        ids=sample_ids
    )
    
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    qa_dataset = []
    
    for point in points:
        chunk_text = point.payload.get("text", "")
        if len(chunk_text.strip()) < 50:
            continue 
            
        system_prompt = "You are a data generator. Given a text snippet, generate exactly one highly specific question that can be answered by the text. Output ONLY the question, nothing else."
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text Snippet:\n{chunk_text}"}
                ],
                temperature=0.3,
            )
            
            question = response.choices[0].message.content.strip()
            print(f"Generated Q: {question}")
            
            qa_dataset.append({
                "question": question,
                "target_chunk_id": point.id
            })
        except Exception as e:
            print(f"Failed to generate question: {e}")
            
    return qa_dataset

def evaluate_retriever(indexer: HybridIndexer, qa_dataset: list, alpha: float, top_k: int) -> dict:
    
    hits = 0
    mrr_sum = 0.0
    
    for item in qa_dataset:
        query = item["question"]
        target_id = item["target_chunk_id"]
        
        
        results = indexer.hybrid_search(query, top_k=top_k, alpha=alpha)

        for rank, res in enumerate(results, 1):
            if res["chunk_id"] == target_id:
                hits += 1
                mrr_sum += (1.0 / rank)
                break 
                
    num_q = len(qa_dataset)
    return {
        "hit_rate": hits / num_q if num_q > 0 else 0,
        "mrr": mrr_sum / num_q if num_q > 0 else 0
    }

if __name__ == "__main__":
    indexer = HybridIndexer()
    test_doc = "Test.md" 
    try:
        indexer.ingest_document(test_doc, chunk_size=400, chunk_overlap=50)
    except FileNotFoundError:
        print(f"Error: Could not find '{test_doc}'. Please update the path.")
        exit(1)

    dataset = generate_synthetic_qa(indexer, NUM_EVAL_QUESTIONS)
    
    if not dataset:
        print("Failed to generate dataset. Check your LLM connection.")
        exit(1)

    print(f"\n--- Running Evaluation (Top-{TOP_K}) ---")
    
    metrics = {
        "Dense Only (a=1.0)": evaluate_retriever(indexer, dataset, alpha=1.0, top_k=TOP_K),
        "Sparse Only (a=0.0)": evaluate_retriever(indexer, dataset, alpha=0.0, top_k=TOP_K),
        "Hybrid RRF (a=0.5)": evaluate_retriever(indexer, dataset, alpha=0.5, top_k=TOP_K)
    }
    
    print("\n" + "="*50)
    print(f"{'Method':<20} | {'Hit Rate':<12} | {'MRR':<12}")
    print("-" * 50)
    for method, scores in metrics.items():
        print(f"{method:<20} | {scores['hit_rate']:<12.2f} | {scores['mrr']:<12.2f}")
    print("="*50)