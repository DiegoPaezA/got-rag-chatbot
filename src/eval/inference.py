import json
import os
import time
from tqdm import tqdm
from src.rag.graph_search import GraphSearcher  # Importa tu clase principal

def run_inference():
    input_path = "data/eval/golden_dataset_150.jsonl"
    output_path = "data/eval/predictions.jsonl"
    
    if not os.path.exists(input_path):
        print("❌ Ejecuta el paso 1 primero.")
        return

    # 1. Inicializar tu RAG
    print("⚙️ Inicializando RAG...")
    searcher = GraphSearcher() 

    results = []
    
    # 2. Cargar preguntas
    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    print(f"🚀 Ejecutando inferencia para {len(questions)} preguntas...")
    
    # 3. Loop de inferencia
    for q in tqdm(questions):
        try:
            # Llamada a tu chatbot
            response = searcher.search(q['question']) 
            
            # Asumiendo que response devuelve un dict o string. Ajusta según tu código.
            # Si search() devuelve solo texto: answer_text = response
            # Si devuelve dict: answer_text = response['answer']
            answer_text = response if isinstance(response, str) else str(response)

            result_entry = {
                "custom_id": q['question_id'], # Necesario para Batch API
                "question": q['question'],
                "ground_truth": q['ground_truth'],
                "prediction": answer_text
            }
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error en {q['question_id']}: {e}")
            
    # 4. Guardar predicciones
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"✅ Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    run_inference()