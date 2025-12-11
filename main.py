import argparse
import logging
import os
import sys

# Aseguramos que Python encuentre nuestros módulos en la carpeta 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports de nuestros módulos modulares
from src.ingestion.scraper import FandomScraper, ScraperConfig
from src.graph.builder import GraphBuilder
from src.graph.validator import GraphValidator
from src.graph.edge_builder import EdgeBuilder

# Configuración de logs para ver el progreso en la terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
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
    parser_build = subparsers.add_parser("build", help="Build the Knowledge Graph")
    parser_build.add_argument("--use-llm", action="store_true", help="Enable Gemini/LLM validation for better accuracy")
    
    args = parser.parse_args()

    # --- RUTAS DE DATOS ---
    RAW_DATA_PATH = "data/raw/wiki_dump.jsonl"  # Donde guarda el scraper
    PROCESSED_DIR = "data/processed"           # Donde guardan los builders
    CONFIG_PATH = "cfg/config.json"            # Donde están los prompts y settings

    # ==========================================
    # 1. EJECUCIÓN DEL SCRAPER
    # ==========================================
    if args.command == "scrape":
        logger.info("🚀 Starting Scraping Pipeline...")
        
        # Configuramos el scraper
        config = ScraperConfig(
            base_url="https://gameofthrones.fandom.com/api.php",
            user_agent="WesterosBot/2.0 (The Architect)",
            timeout=15
        )
        
        scraper = FandomScraper(config)
        
        # Ejecutamos (el scraper maneja internamente el modo 'append' si se corta)
        scraper.run(output_path=RAW_DATA_PATH)

    # ==========================================
    # 2. CONSTRUCCIÓN DEL GRAFO
    # ==========================================
    elif args.command == "build":
        logger.info("🏗️  Starting Graph Construction Pipeline...")
        
        # --- PASO 1: Extracción de Nodos (Heurística) ---
        # Convierte el texto crudo en entidades JSONL preliminares
        logger.info("--- [STEP 1/3] Node Extraction (Heuristic) ---")
        builder = GraphBuilder(input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR)
        builder.build()
        
        # --- PASO 2: Validación con IA (Opcional) ---
        # Si activas --use-llm, Gemini revisará los nodos con baja confianza
        if args.use_llm:
            logger.info("--- [STEP 2/3] LLM Validation (Gemini) ---")
            
            # Instanciamos el validador pasándole la config externa
            validator = GraphValidator(
                data_dir=PROCESSED_DIR, 
                config_path=CONFIG_PATH
            )
            validator.validate()
        else:
            logger.info("--- [STEP 2/3] Skipping LLM Validation (using heuristics only) ---")

        # --- PASO 3: Construcción de Aristas (Schema-Aware) ---
        # Conecta los nodos usando reglas lógicas, respetando los tipos validados
        logger.info("--- [STEP 3/3] Edge Generation (Schema-Aware) ---")
        edge_builder = EdgeBuilder(data_dir=PROCESSED_DIR)
        edge_builder.run()
        
        logger.info("✅ Pipeline Finished Successfully! Data ready in 'data/processed/'")

if __name__ == "__main__":
    main()