import argparse
import logging
import os
import sys

# Aseguramos que Python encuentre nuestros módulos en la carpeta 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports de nuestros módulos modulares
from src.ingestion.scraper import FandomScraper, ScraperConfig
from src.ingestion.neo4j_loader import Neo4jLoader
from src.graph.builder import GraphBuilder
from src.graph.validator import GraphValidator
from src.graph.edge_builder import EdgeBuilder
from src.rag.vector_store import VectorDBBuilder  # <--- NUEVO IMPORT

# 1. Configuración global de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

# 2. SILENCIADOR DE RUIDO EXTERNO (Vital para ver tqdm limpio)
# Evita que httpx y google impriman cada petición POST
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR) # Chroma suele ser ruidoso con telemetría

logger = logging.getLogger("Orchestrator")

def main():
    # Definición de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Game of Thrones RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- COMANDO: SCRAPE ---
    # Ejemplo: python main.py scrape
    parser_scrape = subparsers.add_parser("scrape", help="Download raw data from Wiki")
    parser_scrape.add_argument("--limit", type=int, help="Limit number of articles (useful for quick testing)")
    
    # --- COMANDO: BUILD ---
    # Ejemplo: python main.py build --use-llm
    parser_build = subparsers.add_parser("build", help="Build the Knowledge Graph (Nodes + Edges)")
    parser_build.add_argument("--use-llm", action="store_true", help="Enable Gemini/LLM validation for better accuracy")
    
    # --- COMANDO: EMBED (NUEVO) ---
    # Ejemplo: python main.py embed --reset
    parser_embed = subparsers.add_parser("embed", help="Create Vector Database (ChromaDB)")
    parser_embed.add_argument("--reset", action="store_true", help="Delete existing DB and rebuild from scratch")
    
    # --- COMANDO: LOAD-GRAPH ---
    parser_load = subparsers.add_parser("load-graph", help="Load processed data into Neo4j Database")

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
        
        config = ScraperConfig(
            base_url="https://gameofthrones.fandom.com/api.php",
            user_agent="WesterosBot/2.0 (The Architect)",
            timeout=15
        )
        
        scraper = FandomScraper(config)
        scraper.run(output_path=RAW_DATA_PATH)

    # ==========================================
    # 2. CONSTRUCCIÓN DEL GRAFO
    # ==========================================
    elif args.command == "build":
        logger.info("🏗️  Starting Graph Construction Pipeline...")
        
        # --- PASO 1: Extracción de Nodos (Heurística) ---
        logger.info("--- [STEP 1/3] Node Extraction (Heuristic) ---")
        builder = GraphBuilder(input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR)
        builder.build()
        
        # --- PASO 2: Validación con IA (Opcional) ---
        if args.use_llm:
            logger.info("--- [STEP 2/3] LLM Validation (Gemini) ---")
            validator = GraphValidator(
                data_dir=PROCESSED_DIR, 
                config_path=CONFIG_PATH
            )
            validator.validate()
        else:
            logger.info("--- [STEP 2/3] Skipping LLM Validation (using heuristics only) ---")

        # --- PASO 3: Construcción de Aristas (Schema-Aware) ---
        logger.info("--- [STEP 3/3] Edge Generation (Schema-Aware) ---")
        edge_builder = EdgeBuilder(data_dir=PROCESSED_DIR)
        edge_builder.run()
        
        logger.info("✅ Graph Build Finished! Nodes and Edges ready in 'data/processed/'")

    # ==========================================
    # 3. BASE DE DATOS VECTORIAL (EMBEDDINGS)
    # ==========================================
    elif args.command == "embed":
        logger.info("🧠 Starting Vector Database Generation...")
        
        # Verificación de seguridad
        docs_file = os.path.join(PROCESSED_DIR, "documents.jsonl")
        if not os.path.exists(docs_file):
            logger.error(f"❌ Documents file not found at {docs_file}. Run 'python main.py build' first.")
            return

        try:
            vector_builder = VectorDBBuilder(data_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
            vector_builder.build(reset=args.reset)
            logger.info("✅ Vector DB Ready! You can now query the RAG system.")
        except Exception as e:
            logger.error(f"❌ Failed to build Vector DB: {e}")
            
    # ==========================================
    # 3. CARGA A NEO4J (GRAFO)
    # ==========================================
    elif args.command == "load-graph":
        logger.info("🚀 Starting Neo4j Ingestion Pipeline...")
        
        # Verificación rápida
        if not os.path.exists(os.path.join(PROCESSED_DIR, "nodes.jsonl")) and not os.path.exists(os.path.join(PROCESSED_DIR, "nodes_validated.jsonl")):
             logger.error("❌ No processed data found. Run 'build' first.")
             return

        try:
            loader = Neo4jLoader(data_dir=PROCESSED_DIR)
            loader.run()
        except Exception as e:
            logger.error(f"Failed to load Neo4j: {e}")
            logger.warning("💡 Tip: Ensure Docker container is running: 'docker-compose up -d'")

if __name__ == "__main__":
    main()