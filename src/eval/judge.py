import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt del Juez
JUDGE_PROMPT = """You are an impartial judge evaluating a chatbot's answers about Game of Thrones.
You will be given a QUESTION, a GROUND TRUTH (correct answer), and a PREDICTION (chatbot answer).

Your task is to determine if the PREDICTION contains the semantic meaning of the GROUND TRUTH.
- Be lenient with phrasing (e.g., "Jon Snow" == "Jon").
- Be strict with facts (e.g., "Ned Stark" != "Robb Stark").
- If the prediction says "I don't know" or retrieves wrong info, Score is 0.

Return ONLY a JSON object:
{"score": 1 if correct else 0, "reason": "short explanation"}
"""

def prepare_batch_file(predictions_path, batch_input_path):
    """Convierte predicciones al formato JSONL que exige la Batch API de Gemini."""
    batch_requests = []
    
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            
            # Construir el prompt para este caso específico
            user_content = f"""
            QUESTION: {entry['question']}
            GROUND TRUTH: {entry['ground_truth']}
            PREDICTION: {entry['prediction']}
            """
            
            # Formato específico de Batch request
            request = {
                "custom_id": entry['custom_id'],
                "method": "generateContent",
                "params": {
                    "model": "models/gemini-1.5-flash", # Modelo barato y rápido
                    "content": {
                        "role": "user",
                        "parts": [{"text": JUDGE_PROMPT + "\n\n" + user_content}]
                    },
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }
            }
            batch_requests.append(request)

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    
    return len(batch_requests)

def run_batch_evaluation():
    pred_path = "data/eval/predictions.jsonl"
    batch_jsonl = "data/eval/batch_requests.jsonl"
    results_path = "data/eval/evaluation_results.jsonl"

    if not os.path.exists(pred_path):
        print("❌ No hay predicciones. Ejecuta paso 2.")
        return

    # 1. Preparar archivo
    count = prepare_batch_file(pred_path, batch_jsonl)
    print(f"📦 Preparado lote de {count} solicitudes.")

    # 2. Subir archivo a Gemini File API
    batch_file = genai.upload_file(batch_jsonl)
    print(f"☁️ Archivo subido: {batch_file.name}")

    # 3. Crear el Job
    print("🚀 Enviando Batch Job a Google...")
    batch_job = genai.create_batch_job(
        display_name="got_chatbot_eval_01",
        model="models/gemini-1.5-flash",
        source=batch_file.name,
        dest_format="jsonl"
    )
    
    print(f"Job creado: {batch_job.name}. Estado: {batch_job.state}")
    print("⏳ Esperando completitud (esto puede tomar varios minutos)...")

    # 4. Polling (esperar resultado)
    while batch_job.state.name == "ACTIVE" or batch_job.state.name == "PROCESSING":
        time.sleep(30) # Revisar cada 30 segundos
        batch_job = genai.get_batch_job(batch_job.name)
        print(f"Estado actual: {batch_job.state.name}...")

    if batch_job.state.name == "FAILED":
        print(f"❌ El Batch Job falló: {batch_job.error}")
        return

    # 5. Descargar resultados
    print("✅ Job completado. Descargando resultados...")
    output_file_name = batch_job.output_file
    # El SDK no descarga directo, leemos el contenido
    # Nota: A veces tarda un poco en estar disponible para lectura
    
    # Una forma robusta es listar los archivos o re-obtener referencia
    # Para simplificar en este script de ejemplo:
    content = genai.get_file(output_file_name).download() # Depende de la versión del SDK
    # Si la versión de SDK es antigua, puede requerir requests normal. 
    # Asumimos google-generativeai actualizado.
    
    # Parsear y Guardar
    # El contenido viene en bytes, decodificamos
    decoded_content = content.decode('utf-8')
    
    total_score = 0
    total_items = 0
    
    with open(results_path, "w", encoding="utf-8") as f:
        for line in decoded_content.strip().split('\n'):
            res = json.loads(line)
            # Extraer la respuesta del modelo
            try:
                model_output = res['response']['candidates'][0]['content']['parts'][0]['text']
                evaluation = json.loads(model_output)
                
                # Unir con el ID original para saber cual pregunta fue
                custom_id = res['custom_id']
                
                final_record = {
                    "id": custom_id,
                    "score": evaluation.get("score", 0),
                    "reason": evaluation.get("reason", "N/A")
                }
                f.write(json.dumps(final_record) + "\n")
                
                total_score += final_record['score']
                total_items += 1
            except Exception as e:
                print(f"Error parseando linea de resultado: {e}")

    print(f"\n📊 RESULTADOS FINALES ({total_items} items):")
    print(f"🎯 Precisión (Accuracy): {total_score / total_items:.2%}")
    print(f"💾 Detalle guardado en: {results_path}")

if __name__ == "__main__":
    run_batch_evaluation()