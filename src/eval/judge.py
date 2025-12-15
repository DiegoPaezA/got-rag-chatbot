import json
import os
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Nota: Usamos StrOutputParser para evitar errores de parseo JSON estricto
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

class LLMJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY no encontrada en .env")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0, # Determinista
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
        - If the prediction retrieves wrong info or extra wrong entities, Score is 0.
        
        OUTPUT FORMAT:
        You must return ONLY a raw JSON object. Do not use Markdown code blocks. Do not add explanations outside the JSON.
        
        CORRECT EXAMPLE:
        {{
            "score": 1,
            "reason": "The prediction matches the ground truth accurately."
        }}

        INCORRECT EXAMPLE (Do NOT do this):
        Score: 0
        Reason: The answer is wrong because...
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
        
        return prompt | self.llm | StrOutputParser()

    def _robust_parse(self, text_output: str) -> Dict[str, Any]:
        """
        Intenta extraer JSON de respuestas sucias o mal formadas.
        Si falla el JSON, usa Regex para buscar patrones como 'Score: 0'.
        """
        text = text_output.strip()
        
        # 1. Intentar limpiar bloques de código markdown
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Intentar parsear JSON directo
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass # Falló JSON puro, vamos al plan B

        # 3. Plan B: Extracción Heurística con Regex
        score_match = re.search(r'(?:score|Score)["\s:]+([01])', text)
        score = int(score_match.group(1)) if score_match else 0
        
        reason = text.replace('"', '').replace('{', '').replace('}', '').strip()
        
        # Si falló el JSON, es bueno saberlo, pero no spamear logs si el regex lo salvó
        # logger.warning(f"⚠️ JSON inválido rescatado con Regex. Score: {score}")
        return {"score": score, "reason": reason}

    def evaluate(self, predictions_path: str, results_path: str):
        if not os.path.exists(predictions_path):
            logger.error(f"❌ No se encontró {predictions_path}")
            return

        entries = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        logger.info(f"👨‍⚖️ Iniciando evaluación de {len(entries)} respuestas...")

        results = []
        total_score = 0
        valid_items = 0

        for entry in tqdm(entries, desc="Evaluando"):
            try:
                # Invocación al LLM
                raw_response = self.chain.invoke({
                    "question": entry.get("question"),
                    "ground_truth": entry.get("ground_truth"),
                    "prediction": entry.get("prediction")
                })
                
                # Parseo Robusto
                parsed_response = self._robust_parse(raw_response)
                
                final_record = {
                    "id": entry.get("custom_id"),
                    "question": entry.get("question"),
                    "prediction": entry.get("prediction"),
                    "ground_truth": entry.get("ground_truth"),
                    "score": parsed_response.get("score", 0),
                    "reason": parsed_response.get("reason", "N/A"),
                    "type": entry.get("type", "Unknown"),
                    "evidence_source": entry.get("evidence_source")
                }
                
                results.append(final_record)
                total_score += final_record["score"]
                valid_items += 1

            except Exception as e:
                logger.error(f"❌ Error fatal evaluando ID {entry.get('custom_id')}: {e}")
                continue

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        accuracy = (total_score / valid_items) * 100 if valid_items > 0 else 0
        logger.info(f"\n📊 RESULTADOS FINALES ({valid_items} items):")
        logger.info(f"🎯 Precisión Global: {accuracy:.2f}%")
        logger.info(f"💾 Resultados: {results_path}")
        
        self._print_analytics(results)

    def _print_analytics(self, results):
        from collections import defaultdict
        by_source = defaultdict(list)
        for r in results:
            src = r.get("evidence_source", "Unknown")
            by_source[src].append(r["score"])
            
        print("\n📈 Desglose por Fuente:")
        for source, scores in by_source.items():
            avg = (sum(scores) / len(scores)) * 100 if scores else 0
            print(f"   - {source}: {avg:.2f}% ({len(scores)} preguntas)")