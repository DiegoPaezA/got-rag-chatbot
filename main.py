from src.ingestion.scraper import FandomScraper, ScraperConfig

def main():
    # 1. Configurar
    config = ScraperConfig(
        base_url="https://gameofthrones.fandom.com/api.php",
        user_agent="MiBotEducativo/1.0"
    )
    
    # 2. Instanciar
    scraper = FandomScraper(config)
    
    # 3. Ejecutar
    scraper.run(output_path="data/raw/wiki_dump.jsonl")

if __name__ == "__main__":
    main()