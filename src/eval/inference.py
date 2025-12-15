import json
import os
import logging
from tqdm import tqdm

from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Silenciamos ruido externo
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def run_inference():
    # Rutas
    input_path = "data/eval/golden_dataset_150.jsonl"
    output_path = "data/eval/predictions.jsonl"
    
    if not os.path.exists(input_path):
        logger.error(f"❌ No se encontró {input_path}. Ejecuta el paso de generación primero.")
        return

    # 1. Inicializar el Sistema Real
    logger.info("⚙️ Inicializando componentes del RAG (Retriever + Generator)...")
    try:
        retriever = HybridRetriever()
        generator = RAGGenerator()
    except Exception as e:
        logger.error(f"❌ Error inicializando el RAG: {e}")
        return

    results = []
    
    # 2. Cargar preguntas del dataset
    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    logger.info(f"🚀 Ejecutando inferencia REAL para {len(questions)} preguntas...")
    
    # 3. Bucle de Inferencia
    for q in tqdm(questions, desc="Procesando preguntas"):
        try:
            query = q['question']
            
            # A. Recuperación (Usa tu lógica de src/rag/retriever.py)
            # Esto busca en Grafo Y Vectores
            context = retriever.retrieve(query)
            
            # B. Generación (Usa tu lógica de src/rag/generator.py)
            # Esto usa el LLM para sintetizar la respuesta
            prediction = generator.generate_answer(query, context)
            ground_truth = q.get('ground_truth', q.get('answer', 'N/A'))
            # C. Guardar resultado
            result_entry = {
                "custom_id": q.get('question_id'),
                "question": query,
                "ground_truth": ground_truth,
                "prediction": prediction,
                # Metadata para analytics posterior
                "type": q.get('type', 'Unknown'),         
                "evidence_source": q.get('evidence_source', 'Unknown')
            }
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"⚠️ Error en pregunta ID {q.get('question_id')}: {e}")
            # Guardamos un fallo controlado para no romper el lote
            results.append({
                "custom_id": q.get('question_id'),
                "question": q['question'],
                "ground_truth": q['ground_truth'],
                "prediction": "ERROR_DURING_INFERENCE",
                "error": str(e)
            })
            
    # 4. Guardar predicciones en JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    logger.info(f"✅ Predicciones guardadas exitosamente en: {output_path}")

if __name__ == "__main__":
    run_inference()