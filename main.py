import argparse
import logging
import os
import sys

# Ensure Python finds our modules in the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modular components
from src.utils.logger import setup_logging
from src.ingestion.scraper import FandomScraper, ScraperConfig
from src.ingestion.neo4j_loader import Neo4jLoader
from src.graph.builder import GraphBuilder
from src.graph.validator import GraphValidator
from src.graph.cleaner import GraphCleaner
from src.graph.edge_builder import EdgeBuilder
from src.rag.vector_store import VectorDBBuilder


from src.eval.dataset_generator import StratifiedDatasetGenerator
from src.eval.inference import run_inference
from src.eval.judge import run_batch_evaluation

# Initialize centralized logging system
setup_logging()

# Silence external library noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

logger = logging.getLogger("Orchestrator")

def main():
    # Definición de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Game of Thrones RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- COMANDO: SCRAPE ---
    parser_scrape = subparsers.add_parser("scrape", help="Download raw data from Wiki")
    parser_scrape.add_argument("--limit", type=int, help="Limit number of articles")
    
    # --- COMANDO: BUILD ---
    parser_build = subparsers.add_parser("build", help="Build the Knowledge Graph")
    parser_build.add_argument("--use-llm", action="store_true", help="Enable Gemini validation")
    parser_build.add_argument("--clean-llm", action="store_true", help="Enable Gemini cleaning")
    
    # --- COMANDO: EMBED ---
    parser_embed = subparsers.add_parser("embed", help="Create Vector Database")
    parser_embed.add_argument("--reset", action="store_true", help="Rebuild DB from scratch")
    
    # --- COMANDO: LOAD-GRAPH ---
    parser_load = subparsers.add_parser("load-graph", help="Load processed data into Neo4j")

    # --- COMANDO: EVAL (NUEVO) ---
    parser_eval = subparsers.add_parser("eval", help="Run LLM-as-a-Judge Evaluation Pipeline")
    parser_eval.add_argument("--step", choices=["generate", "infer", "judge", "all"], default="all", help="Which evaluation step to run")
    parser_eval.add_argument("--limit", type=int, default=150, help="Number of questions to generate (default: 150)")

    args = parser.parse_args()

    # --- RUTAS DE DATOS ---
    RAW_DATA_PATH = "data/raw/wiki_dump.jsonl"
    PROCESSED_DIR = "data/processed"
    CONFIG_PATH = "cfg/config.json"

    # ==========================================
    # 1. EJECUCIÓN DEL SCRAPER
    # ==========================================
    if args.command == "scrape":
        logger.info("🚀 Starting Scraping Pipeline...")
        config = ScraperConfig(base_url="https://gameofthrones.fandom.com/api.php", user_agent="WesterosBot/2.0", timeout=15)
        scraper = FandomScraper(config)
        scraper.run(output_path=RAW_DATA_PATH)

    # ==========================================
    # 2. CONSTRUCCIÓN DEL GRAFO
    # ==========================================
    elif args.command == "build":
        logger.info("🏗️  Starting Graph Construction Pipeline...")
        
        # Step 1: Extraction
        logger.info("--- [STEP 1/3] Node Extraction (Heuristic) ---")
        builder = GraphBuilder(input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR)
        builder.build()
        
        # Step 2: Validation
        if args.use_llm:
            logger.info("--- [STEP 2/3] LLM Validation (Gemini) ---")
            validator = GraphValidator(data_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
            validator.validate()
        else:
            logger.info("--- [STEP 2/3] Skipping LLM Validation ---")

        # Step 2.5: Cleaning
        if args.clean_llm:
            logger.info("--- [STEP 2.5/4] LLM Cleaning & Normalization ---")
            cleaner = GraphCleaner(data_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
            cleaner.run()
        else:
            logger.info("--- [STEP 2.5/4] Skipping LLM Cleaning ---")
        
        # Step 3: Edge Building
        logger.info("--- [STEP 3/3] Edge Generation (Schema-Aware) ---")
        edge_builder = EdgeBuilder(data_dir=PROCESSED_DIR)
        edge_builder.run()
        
        logger.info("✅ Graph Build Finished!")

    # ==========================================
    # 3. BASE DE DATOS VECTORIAL
    # ==========================================
    elif args.command == "embed":
        logger.info("🧠 Starting Vector Database Generation...")
        docs_file = os.path.join(PROCESSED_DIR, "documents.jsonl")
        if not os.path.exists(docs_file):
            logger.error(f"❌ Documents file not found. Run 'build' first.")
            return
        try:
            vector_builder = VectorDBBuilder(data_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
            vector_builder.build(reset=args.reset)
            logger.info("✅ Vector DB Ready!")
        except Exception as e:
            logger.error(f"❌ Failed to build Vector DB: {e}")

    # ==========================================
    # 4. CARGA A NEO4J
    # ==========================================
    elif args.command == "load-graph":
        logger.info("🚀 Starting Neo4j Ingestion Pipeline...")
        if not os.path.exists(os.path.join(PROCESSED_DIR, "nodes.jsonl")):
             logger.error("❌ No processed data found. Run 'build' first.")
             return
        try:
            loader = Neo4jLoader(data_dir=PROCESSED_DIR)
            loader.run()
        except Exception as e:
            logger.error(f"Failed to load Neo4j: {e}")

    # ==========================================
    # 5. EVALUACIÓN (LLM-AS-A-JUDGE)
    # ==========================================
    elif args.command == "eval":
        logger.info("🧪 Starting Evaluation Pipeline (LLM-as-a-Judge)...")
        
        # --- PASO 1: GENERACIÓN ---
        if args.step in ["generate", "all"]:
            logger.info(f"--- [EVAL 1/3] Generating Golden Dataset ({args.limit} questions) ---")
            try:
                generator = StratifiedDatasetGenerator(data_dir=PROCESSED_DIR)
                generator.generate(num_questions=args.limit)
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                return

        # --- PASO 2: INFERENCIA ---
        if args.step in ["infer", "all"]:
            logger.info("--- [EVAL 2/3] Running Inference (Chatbot Responses) ---")
            try:
                # Asegúrate de que inference.py tenga una función main() o run_inference() exportable
                run_inference() 
            except Exception as e:
                logger.error(f"❌ Inference failed: {e}")
                return

        # --- PASO 3: JUEZ (BATCH API) ---
        if args.step in ["judge", "all"]:
            logger.info("--- [EVAL 3/3] Running LLM Judge (Batch API) ---")
            try:
                run_batch_evaluation()
            except Exception as e:
                logger.error(f"❌ Judge execution failed: {e}")
                return
        
        logger.info("✅ Evaluation Pipeline Triggered Successfully!")

if __name__ == "__main__":
    main()