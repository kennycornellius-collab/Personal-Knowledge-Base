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
RANDOM_SEED = 42 

def generate_synthetic_qa(indexer: HybridIndexer, num_questions: int) -> list:
    print(f"\n--- Generating {num_questions} Synthetic Q&A Pairs ---")
    random.seed(RANDOM_SEED)
    
    all_uuids = indexer.bm25_id_map
    shuffled_ids = all_uuids.copy()
    random.shuffle(shuffled_ids)
    
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    qa_dataset = []
    candidate_ids = shuffled_ids[:min(num_questions * 3, len(shuffled_ids))]
    
    candidate_points = indexer.qdrant.retrieve(
        collection_name=indexer.collection_name, 
        ids=candidate_ids
    )
    
    for point in candidate_points:
        if len(qa_dataset) >= num_questions:
            break
            
        chunk_text = point.payload.get("text", "")
        if len(chunk_text.strip()) < 50:
            continue 
            
        system_prompt = """You are an evaluation data generator. Given a text snippet, generate exactly one question that:
        1. Tests the MEANING or CONCEPT in the text, not surface keywords
        2. Could be asked naturally by someone who does NOT have the text in front of them
        3. Uses different vocabulary than the source text where possible
        Output ONLY the question, nothing else."""
        
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
            print(f"[{len(qa_dataset)+1}/{num_questions}] Generated Q: {question}")
            
            qa_dataset.append({
                "question": question,
                "target_chunk_id": point.id 
            })
        except Exception as e:
            print(f"Failed to generate question: {e}")
            
    print(f"\nGenerated {len(qa_dataset)} valid questions (requested {num_questions})")
    return qa_dataset

def evaluate_retriever(indexer: HybridIndexer, qa_dataset: list, alpha: float, top_k: int, method_name: str) -> dict:
    hits = 0
    mrr_sum = 0.0
    num_q = len(qa_dataset)
    
    print(f"\nEvaluating: {method_name} (Alpha={alpha})")
    
    for i, item in enumerate(qa_dataset, 1):
        query = item["question"]
        target_id = item["target_chunk_id"]
        
        print(f"  [{i}/{num_q}] Query: {query[:50]}")
        
        results = indexer.hybrid_search(query, top_k=top_k, alpha=alpha)
        
        for rank, res in enumerate(results, 1):
            if res["chunk_id"] == target_id:
                hits += 1
                mrr_sum += (1.0 / rank)
                break 
                
    return {
        "hit_rate": hits / num_q if num_q > 0 else 0,
        "mrr": mrr_sum / num_q if num_q > 0 else 0
    }

if __name__ == "__main__":
    indexer = HybridIndexer()
    test_doc = "README.md" 
    
    try:
        indexer.ingest_document(test_doc, chunk_size=400, chunk_overlap=50)
    except FileNotFoundError:
        print(f"Error: Could not find '{test_doc}'. Please update the path.")
        exit(1)

    dataset = generate_synthetic_qa(indexer, NUM_EVAL_QUESTIONS)
    
    if not dataset:
        print("Failed to generate dataset. Check your LLM connection.")
        exit(1)

    print(f"\n--- Running Evaluation (Top-{TOP_K} on {len(dataset)} Questions) ---")
    
    metrics = {
        "Dense Only": evaluate_retriever(indexer, dataset, alpha=1.0, top_k=TOP_K, method_name="Dense Only"),
        "Sparse Only": evaluate_retriever(indexer, dataset, alpha=0.0, top_k=TOP_K, method_name="Sparse Only"),
        "Hybrid RRF": evaluate_retriever(indexer, dataset, alpha=0.5, top_k=TOP_K, method_name="Hybrid RRF")
    }
    
    print("\n" + "="*60)
    print(f"{'Method':<15} | {'Hit Rate':<10} | {'MRR':<10}")
    print("-" * 60)
    for method, scores in metrics.items():
        print(f"{method:<15} | {scores['hit_rate']:<10.2f} | {scores['mrr']:<10.2f}")
    print("="*60)
    print("* MRR: Average of 1/rank for correct hits (1.0 = always rank 1, 0.5 = always rank 2).")