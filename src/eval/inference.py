import json
import os
import logging
from tqdm import tqdm

from src.rag.retriever import HybridRetriever
from src.rag.augmenter import ContextAugmenter
from src.rag.generator import RAGGenerator

# Local logging configuration and noise reduction
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def run_inference():
    """Run inference over the golden dataset and save predictions to JSONL."""
    # Paths
    input_path = "data/eval/golden_dataset_150.jsonl"
    output_path = "data/eval/predictions.jsonl"
    
    if not os.path.exists(input_path):
        logger.error(f"❌ Input not found: {input_path}. Run generation first.")
        return

    # 1) Initialize the RAG pipeline components
    logger.info("⚙️ Initializing RAG components (Retriever -> Augmenter -> Generator)...")
    try:
        retriever = HybridRetriever()
        augmenter = ContextAugmenter()
        generator = RAGGenerator()
    except Exception as e:
        logger.error(f"❌ Error initializing RAG: {e}")
        return

    results = []
    
    # 2) Load questions from dataset
    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    logger.info(f"🚀 Running real inference for {len(questions)} questions...")
    
    # 3) Inference loop
    for q in tqdm(questions, desc="Processing questions"):
        try:
            query = q['question']
            
            # A) Retrieval (raw data from graph/vector stores)
            raw_context_data = retriever.retrieve(query)
            
            # B) Augmentation (processing and formatting for the LLM)
            # Augmenter converts raw context into LLM-ready text
            narrative_context = augmenter.build_context(
                query=query,
                vector_context=raw_context_data.get("vector_context", []),
                graph_context=raw_context_data.get("graph_context", [])
                )
        
            
            # C) Generation (LLM synthesis)
            # Generator consumes already-refined context
            prediction = generator.generate_answer(query, narrative_context)
            
            ground_truth = q.get('ground_truth', q.get('answer', 'N/A'))
            
            # D) Save result
            result_entry = {
                "custom_id": q.get('question_id'),
                "question": query,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "retrieved_context": narrative_context, 
                "type": q.get('type', 'Unknown'),         
                "evidence_source": q.get('evidence_source', 'Unknown')
            }
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"⚠️ Error on question ID {q.get('question_id')}: {e}")
            # Save a controlled failure to avoid breaking the batch
            results.append({
                "custom_id": q.get('question_id'),
                "question": q['question'],
                "ground_truth": q.get('ground_truth', 'N/A'),
                "prediction": "ERROR_DURING_INFERENCE",
                "error": str(e)
            })
            
    # 4) Save predictions in JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    logger.info(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    run_inference()