import argparse
import logging
import os
import sys

# Ensure Python can import modules from the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modular components
from src.utils.logger import setup_logging
from src.config_manager import ConfigManager
from src.ingestion.scraper import FandomScraper, ScraperConfig
from src.ingestion.neo4j_loader import Neo4jLoader
from src.graph.builder import GraphBuilder
from src.graph.validator import GraphValidator
from src.graph.cleaner import GraphCleaner
from src.graph.edge_builder import EdgeBuilder
from src.rag.vector_store import VectorDBBuilder


from src.eval.dataset_generator import Neo4jDatasetGenerator
from src.eval.inference import run_inference
from src.eval.judge import RAGJudge

# Initialize centralized logging system
setup_logging()

# Load configuration
ConfigManager.load("cfg/config.json")

# Silence external library noise
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("google.generativeai").setLevel(logging.WARNING)
# logging.getLogger("grpc").setLevel(logging.WARNING)
# logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

logger = logging.getLogger("Orchestrator")

def main():
    """CLI entrypoint orchestrating the end-to-end RAG pipeline.

    Provides subcommands to scrape data, build the graph, create embeddings,
    load Neo4j, and run the evaluation pipeline (generate → infer → judge).
    """
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Game of Thrones RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- COMMAND: SCRAPE ---
    parser_scrape = subparsers.add_parser("scrape", help="Download raw data from Wiki")
    parser_scrape.add_argument("--limit", type=int, help="Limit number of articles")
    
    # --- COMMAND: BUILD ---
    parser_build = subparsers.add_parser("build", help="Build the Knowledge Graph")
    parser_build.add_argument("--use-heuristic", action="store_true", help="Use heuristic extraction")
    parser_build.add_argument("--use-llm", action="store_true", help="Enable Gemini validation")
    parser_build.add_argument("--clean-llm", action="store_true", help="Enable Gemini cleaning")
    
    # --- COMMAND: EMBED ---
    parser_embed = subparsers.add_parser("embed", help="Create Vector Database")
    parser_embed.add_argument("--reset", action="store_true", help="Rebuild DB from scratch")
    
    # --- COMMAND: LOAD-GRAPH ---
    parser_load = subparsers.add_parser("load-graph", help="Load processed data into Neo4j")

    # --- COMMAND: EVAL ---
    parser_eval = subparsers.add_parser("eval", help="Run LLM-as-a-Judge Evaluation Pipeline")
    parser_eval.add_argument("--step", choices=["generate", "infer", "judge", "all"], default="all", help="Which evaluation step to run")
    parser_eval.add_argument("--limit", type=int, default=150, help="Number of questions to generate (default: 150)")

    args = parser.parse_args()

    # --- DATA PATHS (from config) ---
    paths = ConfigManager.get_paths()
    RAW_DATA_PATH = paths.get("raw_data", "data/raw/wiki_dump.jsonl")
    PROCESSED_DIR = paths.get("processed_dir", "data/processed")
    CONFIG_PATH = paths.get("config_path", "cfg/config.json")

    # ==========================================
    # 1. SCRAPER EXECUTION
    # ==========================================
    if args.command == "scrape":
        logger.info("🚀 Starting Scraping Pipeline...")
        scraper_cfg = ConfigManager.get_section("scraper")
        scraper_config = ScraperConfig(
            base_url=scraper_cfg.get("base_url", "https://gameofthrones.fandom.com/api.php"),
            user_agent=scraper_cfg.get("user_agent", "WesterosBot/2.0"),
            timeout=scraper_cfg.get("timeout", 15)
        )
        scraper = FandomScraper(scraper_config)
        scraper.run(output_path=RAW_DATA_PATH)

    # ==========================================
    # 2. GRAPH CONSTRUCTION
    # ==========================================
    elif args.command == "build":
        logger.info("🏗️  Starting Graph Construction Pipeline...")
        
        # Step 1: Extraction
        if args.use_heuristic:
            logger.info("--- [STEP 1/3] Node Extraction (Heuristic) ---")
            builder = GraphBuilder(input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR)
            builder.build()
        else:
            logger.info("--- [STEP 1/3] Skipping Heuristic Extraction ---")
        
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
    # 3. VECTOR DATABASE
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
    # 4. NEO4J INGESTION
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
    # 5. EVALUATION (LLM-AS-A-JUDGE)
    # ==========================================
    elif args.command == "eval":
        logger.info("🧪 Starting Evaluation Pipeline (LLM-as-a-Judge)...")
       
        # --- STEP 1: GENERATION ---
        if args.step in ["generate", "all"]:
            logger.info(f"--- [EVAL 1/3] Generating Golden Dataset ({args.limit} questions) ---")
            try:
                generator = Neo4jDatasetGenerator()
                generator.generate(num_questions=args.limit)
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                return

        # --- STEP 2: INFERENCE ---
        if args.step in ["infer", "all"]:
            logger.info("--- [EVAL 2/3] Running Inference (Chatbot Responses) ---")
            try:
                # Ensure inference.py exposes a callable run_inference()
                run_inference() 
            except Exception as e:
                logger.error(f"❌ Inference failed: {e}")
                return

        # --- STEP 3: JUDGE (BATCH API) ---
        if args.step in ["judge", "all"]:
            logger.info("--- [EVAL 3/3] Running LLM Judge (Batch API) ---")
            try:
                judge = RAGJudge()
                judge.evaluate()
            except Exception as e:
                logger.error(f"❌ Judge execution failed: {e}")
                return
        
        logger.info("✅ Evaluation Pipeline Triggered Successfully!")

if __name__ == "__main__":
    main()