import json
import os
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from src.utils.logger import setup_logging
from src.config_manager import ConfigManager

# LangChain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm_factory import LLMFactory

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

class RAGJudge:
    """
    Advanced Judge for RAG Systems. Evaluates three key dimensions:
    1. Context Recall: Was the necessary information retrieved?
    2. Faithfulness: Did the model hallucinate information not present in context?
    3. Answer Correctness: Does the answer match the ground truth?
    """

    def __init__(self, llm: BaseChatModel | None = None, config_path: str = "cfg/config.json"):
        self.config = ConfigManager()
        self.config.load(config_path)
        
        # Load LLM settings
        llm_config = ConfigManager.get_llm_config("judge")
        provider = llm_config.get("provider", "google")
        model_name = llm_config.get("model_name", llm_config.get("model", "gemini-2.5-flash"))
        
        # Deterministic judge LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMFactory.create_llm(llm_config, provider=provider)
        
        self.chain = self._create_chain()

    def _create_chain(self):
        """Creates a multi-metric evaluation chain."""
        
        system_prompt = ConfigManager.get("prompts", "judge", "system")
        human_prompt = ConfigManager.get("prompts", "judge", "human")

        # Verificar que no vengan vacíos (por si acaso)
        if not system_prompt or not human_prompt:
            logger.error("❌ Prompts del Juez no encontrados en config.json")
            raise ValueError("Judge prompts missing")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        return prompt | self.llm | StrOutputParser()

    def _robust_parse(self, text_output: str) -> Dict[str, Any]:
        """Parses LLM output into a dictionary, handling Markdown or format errors."""
        text = text_output.strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback regex if JSON fails
            logger.warning(f"⚠️ JSON Parse failed for: {text[:50]}... Attempting regex.")
            recall = 1 if "context_recall\": 1" in text else 0
            faith = 1 if "faithfulness\": 1" in text else 0
            correct = 1 if "correctness\": 1" in text else 0
            return {
                "context_recall": recall, 
                "faithfulness": faith, 
                "correctness": correct, 
                "reason": "Parsed via Regex fallback"
            }

    def evaluate(self):
        """Main evaluation loop reading from predictions.jsonl."""
        predictions_path = ConfigManager.get("paths", "eval", "predictions")
        results_path = ConfigManager.get("paths", "eval", "results")

        if not predictions_path or not os.path.exists(predictions_path):
            logger.error(f"❌ Predictions file not found: {predictions_path}")
            return

        entries = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        logger.info(f"👨‍⚖️ Evaluating {len(entries)} predictions with Multi-Metric Judge...")
        
        results = []
        metrics_sum = {"context_recall": 0, "faithfulness": 0, "correctness": 0}
        valid_items = 0

        for entry in tqdm(entries, desc="Judging"):
            try:
                # Prepare context string (handle list or string format)
                context_raw = entry.get("retrieved_context", "")
                if isinstance(context_raw, list):
                    context_str = "\n".join(context_raw)
                else:
                    context_str = str(context_raw)

                # Invoke LLM
                response_str = self.chain.invoke({
                    "question": entry.get("question"),
                    "ground_truth": entry.get("ground_truth"),
                    "context": context_str[:15000],  # Truncate to avoid token limits
                    "prediction": entry.get("prediction")
                })

                scores = self._robust_parse(response_str)

                # Build result record
                record = {
                    "id": entry.get("custom_id"),
                    "question": entry.get("question"),
                    "type": entry.get("type"),
                    "source": entry.get("evidence_source"),
                    "metrics": scores,
                    "prediction": entry.get("prediction"),
                    "ground_truth": entry.get("ground_truth")
                }
                
                results.append(record)
                
                # Update running totals
                metrics_sum["context_recall"] += scores.get("context_recall", 0)
                metrics_sum["faithfulness"] += scores.get("faithfulness", 0)
                metrics_sum["correctness"] += scores.get("correctness", 0)
                valid_items += 1

            except Exception as e:
                logger.error(f"❌ Error judging {entry.get('custom_id')}: {e}")

        # Save results
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        self._print_analytics(metrics_sum, valid_items, results)

    def _print_analytics(self, sums, total, results):
        if total == 0:
            logger.warning("No valid results to analyze.")
            return

        logger.info("\n📊 === EVALUATION REPORT ===")
        logger.info(f"Analyzed {total} questions.")
        
        # 1. Global Metrics
        logger.info("\n🔹 GLOBAL METRICS:")
        logger.info(f"   ✅ Answer Correctness: {(sums['correctness']/total)*100:.2f}% (User Perception)")
        logger.info(f"   🔍 Context Recall:     {(sums['context_recall']/total)*100:.2f}% (Retriever Quality)")
        logger.info(f"   🤖 Faithfulness:       {(sums['faithfulness']/total)*100:.2f}% (Hallucination Rate)")

        # 2. Diagnosis Matrix
        retrieval_failures = 0
        generation_failures = 0
        
        for r in results:
            m = r["metrics"]
            # Retrieval failed if context didn't have the answer
            if m["context_recall"] == 0:
                retrieval_failures += 1
            # Generation failed if context HAD answer, but final answer was WRONG
            elif m["context_recall"] == 1 and m["correctness"] == 0:
                generation_failures += 1
        
        logger.info("\n🔹 DIAGNOSIS (Where to improve?):")
        logger.info(f"   📉 Retrieval Failures: {retrieval_failures} items (Graph/Vector didn't find info)")
        logger.info(f"   🧠 Reasoning Failures: {generation_failures} items (LLM had info but failed to answer)")

        logger.info(f"\n💾 Detailed results saved to: {ConfigManager.get('paths', 'eval', 'results')}")

if __name__ == "__main__":
    judge = RAGJudge()
    judge.evaluate()