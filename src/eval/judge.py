import json
import os
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.utils.logger import setup_logging
from src.config_manager import ConfigManager

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Note: StrOutputParser is used to avoid strict JSON parsing errors
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

setup_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

class LLMJudge:
    def __init__(self, model_name: str = None):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY not found in .env")

        # Load configuration
        config_manager = ConfigManager()
        llm_config = config_manager.get_llm_config("judge")
        
        # Use provided model_name or fall back to config
        model = model_name or llm_config.get("model", "gemini-2.5-flash")

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=llm_config.get("temperature", 0),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=llm_config.get("max_retries", 3)
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        # Get prompts from config
        config_manager = ConfigManager()
        system_prompt = config_manager.get("prompts", "judge", "system", default="""
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
        """)

        human_prompt = config_manager.get("prompts", "judge", "human", default="""
        QUESTION: {question}
        GROUND TRUTH: {ground_truth}
        PREDICTION: {prediction}
        """)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        return prompt | self.llm | StrOutputParser()

    def _robust_parse(self, text_output: str) -> Dict[str, Any]:
        """Extract JSON from malformed responses, falling back to regex when needed."""
        text = text_output.strip()
        
        # 1. Remove markdown code fences if present
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # JSON parsing failed, try regex

        # 3. Regex-based fallback
        score_match = re.search(r'(?:score|Score)["\s:]+([01])', text)
        score = int(score_match.group(1)) if score_match else 0
        
        reason = text.replace('"', '').replace('{', '').replace('}', '').strip()
        return {"score": score, "reason": reason}

    def evaluate(self, predictions_path: str, results_path: str):
        if not os.path.exists(predictions_path):
            logger.error(f"❌ Predictions file not found: {predictions_path}")
            return

        entries = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        logger.info(f"👨‍⚖️ Starting evaluation of {len(entries)} responses...")

        results = []
        total_score = 0
        valid_items = 0

        for entry in tqdm(entries, desc="Evaluating"):
            try:
                # LLM invocation
                raw_response = self.chain.invoke({
                    "question": entry.get("question"),
                    "ground_truth": entry.get("ground_truth"),
                    "prediction": entry.get("prediction")
                })
                
                # Robust parse
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
                logger.error(f"❌ Fatal error while evaluating ID {entry.get('custom_id')}: {e}")
                continue

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        accuracy = (total_score / valid_items) * 100 if valid_items > 0 else 0
        logger.info(f"\n📊 FINAL RESULTS ({valid_items} items):")
        logger.info(f"🎯 Overall accuracy: {accuracy:.2f}%")
        logger.info(f"💾 Results written to: {results_path}")
        
        self._print_analytics(results)

    def _print_analytics(self, results):
        from collections import defaultdict
        by_source = defaultdict(list)
        for r in results:
            src = r.get("evidence_source", "Unknown")
            by_source[src].append(r["score"])
            
        print("\n📈 Breakdown by source:")
        for source, scores in by_source.items():
            avg = (sum(scores) / len(scores)) * 100 if scores else 0
            print(f"   - {source}: {avg:.2f}% ({len(scores)} questions)")