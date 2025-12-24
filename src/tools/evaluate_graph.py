import sys
import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Import hack to reach project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logging
from src.rag.graph_search import GraphSearcher
from src.config_manager import ConfigManager
from src.utils.llm_factory import LLMFactory
from langchain_core.prompts import PromptTemplate

setup_logging()
logging.getLogger(__name__).setLevel(logging.ERROR)

TEST_SET_PATH = "data/gold/graph_test_set.jsonl"
OUTPUT_REPORT = "reports/graph_evaluation.csv"

class GraphEvaluator:
    def __init__(self):
        load_dotenv()
        self.searcher = GraphSearcher()

        cfg = ConfigManager()
        cfg.load("cfg/config.json")
        llm_cfg = cfg.get_llm_config("judge")
        provider = llm_cfg.get("provider", "google")
        # Strict judge
        self.judge = LLMFactory.create_llm({
            "model_name": llm_cfg.get("model_name", llm_cfg.get("model", "gemini-2.5-flash")),
            "temperature": llm_cfg.get("temperature", 0.0),
            "max_retries": llm_cfg.get("max_retries", 3)
        }, provider=provider)

    def evaluate_result(self, question, ground_truth, db_result, generated_cypher):
        """Ask the LLM to decide if DB JSON contains the correct answer."""
        # Convertimos el resultado de la BD a string para el prompt
        db_result_str = json.dumps(db_result, ensure_ascii=False)
        
        template = """
        You are evaluating a Graph Database Query system.
        
        QUESTION: {question}
        EXPECTED ANSWER (Ground Truth): {ground_truth}
        
        GENERATED CYPHER: {cypher}
        DATABASE RAW OUTPUT (JSON): {db_result}
        
        Task:
        1. Look at the DATABASE RAW OUTPUT.
        2. Does this JSON contain the information to answer the question correctly matching the Ground Truth?
        3. If the JSON is empty or irrelevant, score is 0.
        
        Output JSON only: {{"score": 1 (Pass) or 0 (Fail), "reason": "Short explanation"}}
        """
        
        prompt = PromptTemplate(template=template, input_variables=["question", "ground_truth", "cypher", "db_result"])
        chain = prompt | self.judge
        
        try:
            res = chain.invoke({
                "question": question, 
                "ground_truth": ground_truth, 
                "cypher": generated_cypher,
                "db_result": db_result_str
            })
            return json.loads(res.content.replace("```json", "").replace("```", "").strip())
        except:
            return {"score": 0, "reason": "Judge Error"}

    def run(self):
        if not os.path.exists(TEST_SET_PATH):
            print("❌ No test set found.")
            return

        test_cases = []
        with open(TEST_SET_PATH, 'r') as f:
            for line in f:
                if line.strip(): test_cases.append(json.loads(line))

        results = []
        print(f"🕷️  Evaluating {len(test_cases)} graph queries...")

        for case in tqdm(test_cases):
            q = case["question"]
            truth = case["ground_truth"]
            
            # 1) Generate Cypher
            cypher = self.searcher.generate_cypher(q)
            
            # 2) Execute and capture possible Cypher syntax errors
            try:
                db_data = self.searcher.run_query(q)
                error = None
            except Exception as e:
                db_data = []
                error = str(e)

            # 3) Judge
            if error:
                score = 0
                reason = f"Cypher Error: {error}"
            elif not db_data:
                score = 0
                reason = "Empty Result from DB"
            else:
                eval_res = self.evaluate_result(q, truth, db_data, cypher)
                score = eval_res["score"]
                reason = eval_res["reason"]

            results.append({
                "Question": q,
                "Type": case.get("type"),
                "Score": score,
                "Reason": reason,
                "Cypher": cypher,
                "DB Output": str(db_data)[:100] + "..."  # Truncate for CSV
            })

        # Reporte
        df = pd.DataFrame(results)
        os.makedirs("reports", exist_ok=True)
        df.to_csv(OUTPUT_REPORT, index=False)
        
        acc = df["Score"].mean() * 100
        print(f"\n📊 Graph Accuracy: {acc:.1f}%")
        print(f"📄 Details saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    evaluator = GraphEvaluator()
    evaluator.run()