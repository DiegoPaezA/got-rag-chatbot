import json
import os
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List

# LangChain Imports (Consistencia con el resto del proyecto)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Silenciar ruido
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

# --- MODELO DE SALIDA (PYDANTIC) ---
class JudgeResult(BaseModel):
    score: int = Field(description="1 if the prediction is semantically correct, 0 otherwise.")
    reason: str = Field(description="A short explanation comparing the prediction to the ground truth.")

class LLMJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY no encontrada en .env")

        # Usamos temperature=0 para que sea un juez estricto y determinista
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0, 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=3
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        system_prompt = """
        You are an impartial judge evaluating a chatbot's answers about Game of Thrones.
        
        INPUT DATA:
        1. QUESTION: The user's query.
        2. GROUND TRUTH: The correct fact.
        3. PREDICTION: The chatbot's response.

        TASK:
        Determine if the PREDICTION contains the semantic meaning of the GROUND TRUTH.
        
        RULES:
        - Be lenient with phrasing (e.g., "Jon Snow" == "Jon").
        - Be strict with facts (e.g., "Ned Stark" != "Robb Stark").
        - If the prediction says "I don't know" or retrieves wrong info, Score is 0.
        - If the prediction adds extra correct info, Score is 1.
        
        Return a JSON with 'score' (1 or 0) and 'reason'.
        """

        human_prompt = """
        QUESTION: {question}
        GROUND TRUTH: {ground_truth}
        PREDICTION: {prediction}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        parser = JsonOutputParser(pydantic_object=JudgeResult)
        
        return prompt | self.llm | parser

    def evaluate(self, predictions_path: str, results_path: str):
        if not os.path.exists(predictions_path):
            logger.error(f"❌ No se encontró {predictions_path}")
            return

        # 1. Cargar Predicciones
        entries = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        logger.info(f"👨‍⚖️ Iniciando evaluación de {len(entries)} respuestas con LangChain...")

        results = []
        total_score = 0
        valid_items = 0

        # 2. Bucle de Evaluación
        for entry in tqdm(entries, desc="Evaluando"):
            try:
                # Invocación directa a LangChain
                response = self.chain.invoke({
                    "question": entry.get("question"),
                    "ground_truth": entry.get("ground_truth"),
                    "prediction": entry.get("prediction")
                })
                
                # Armar resultado
                final_record = {
                    "id": entry.get("custom_id"),
                    "question": entry.get("question"),
                    "prediction": entry.get("prediction"),
                    "ground_truth": entry.get("ground_truth"),
                    "score": response.get("score", 0),
                    "reason": response.get("reason", "N/A"),
                    "type": entry.get("type", "Unknown"),           # Analytics
                    "evidence_source": entry.get("evidence_source") # Analytics
                }
                
                results.append(final_record)
                total_score += final_record["score"]
                valid_items += 1

            except Exception as e:
                logger.error(f"❌ Error evaluando ID {entry.get('custom_id')}: {e}")
                continue

        # 3. Guardar Resultados
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        # 4. Reporte Final
        accuracy = (total_score / valid_items) * 100 if valid_items > 0 else 0
        logger.info(f"\n📊 RESULTADOS FINALES ({valid_items} items evaluados):")
        logger.info(f"🎯 Precisión Global (Accuracy): {accuracy:.2f}%")
        logger.info(f"💾 Resultados detallados en: {results_path}")
        
        # Analytics por Fuente (Graph vs Text)
        self._print_analytics(results)

    def _print_analytics(self, results):
        """Imprime desglose de precisión por tipo de fuente."""
        from collections import defaultdict
        
        by_source = defaultdict(list)
        for r in results:
            src = r.get("evidence_source", "Unknown")
            by_source[src].append(r["score"])
            
        print("\n📈 Desglose por Fuente:")
        for source, scores in by_source.items():
            avg = (sum(scores) / len(scores)) * 100 if scores else 0
            print(f"   - {source}: {avg:.2f}% ({len(scores)} preguntas)")

def run_batch_evaluation():
    # Mantenemos el nombre de la función para compatibilidad con main.py
    judge = LLMJudge()
    judge.evaluate(
        predictions_path="data/eval/predictions.jsonl",
        results_path="data/eval/evaluation_results.jsonl"
    )

if __name__ == "__main__":
    run_batch_evaluation()