import json
import os
import logging
from tqdm import tqdm

from src.rag.retriever import HybridRetriever
from src.rag.augmenter import ContextAugmenter
from src.rag.generator import RAGGenerator
from src.config_manager import ConfigManager

# Local logging configuration and noise reduction
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def run_inference():
    """
    Run inference over the golden dataset defined in config and save predictions.
    
    Loads paths from ConfigManager, initializes the RAG pipeline, and processes
    questions sequentially, saving results to the configured predictions path.
    """
    # 1. Load paths from ConfigManager
    input_path = ConfigManager.get("paths", "eval", "golden_dataset")
    output_path = ConfigManager.get("paths", "eval", "predictions")

    # Fallback defaults if config is missing keys
    if not input_path:
        input_path = "data/eval/golden_dataset_150.jsonl"
        logger.warning(f"⚠️ 'golden_dataset' path not found in config. Using default: {input_path}")
    
    if not output_path:
        output_path = "data/eval/predictions.jsonl"
        logger.warning(f"⚠️ 'predictions' path not found in config. Using default: {output_path}")

    if not os.path.exists(input_path):
        logger.error(f"❌ Input file not found: {input_path}. Please run dataset generation first.")
        return

    # 2. Initialize the RAG pipeline components
    logger.info("⚙️ Initializing RAG components (Retriever -> Augmenter -> Generator)...")
    try:
        retriever = HybridRetriever()
        augmenter = ContextAugmenter()
        generator = RAGGenerator()
    except Exception as e:
        logger.error(f"❌ Error initializing RAG pipeline: {e}")
        return

    results = []
    
    # 3. Load questions from dataset
    logger.info(f"📂 Loading questions from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    logger.info(f"🚀 Running inference for {len(questions)} questions...")
    
    # 4. Inference loop
    for q in tqdm(questions, desc="Processing questions"):
        try:
            query = q['question']
            
            # A) Retrieval
            raw_context_data = retriever.retrieve(query)
            
            # B) Augmentation
            narrative_context = augmenter.build_context(
                query=query,
                vector_context=raw_context_data.get("vector_context", []),
                graph_context=raw_context_data.get("graph_context", [])
            )
        
            # C) Generation
            prediction = generator.generate_answer(query, narrative_context)
            
            ground_truth = q.get('ground_truth', q.get('answer', 'N/A'))
            
            # D) Structure Result
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
            results.append({
                "custom_id": q.get('question_id'),
                "question": q['question'],
                "ground_truth": q.get('ground_truth', 'N/A'),
                "prediction": "ERROR_DURING_INFERENCE",
                "error": str(e)
            })
            
    # 5. Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    logger.info(f"✅ Predictions successfully saved to: {output_path}")

if __name__ == "__main__":
    run_inference()