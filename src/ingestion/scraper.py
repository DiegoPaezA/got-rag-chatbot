import time
import json
import logging
import os
from typing import Generator, Dict, Any, Optional
from dataclasses import dataclass

import requests
import mwparserfromhell
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Immutable configuration for the Scraper"""
    base_url: str
    user_agent: str = 'WesterosBot/5.0 (The Vacuum)'
    timeout: int = 10
    retry_delay: int = 5

class WikiParser:
    """Static class responsible for transforming and cleaning data"""
    
    @staticmethod
    def clean_text(raw_wikitext: str) -> Dict[str, Any]:
        """
        Processes raw wikitext and extracts infoboxes and clean text.
        """
        if not raw_wikitext:
            return {"content": "", "infobox": {}}

        try:
            wikicode = mwparserfromhell.parse(raw_wikitext)
            
            # 1. Extract Infobox
            infobox = {}
            # Common infobox filters in fandom
            target_templates = ['infobox', 'character', 'house', 'episode', 'location']
            
            for template in wikicode.filter_templates():
                if any(t in template.name.lower() for t in target_templates):
                    for param in template.params:
                        # Clean key and value
                        key = str(param.name).strip()
                        value = param.value.strip_code().strip()
                        if key and value:
                            infobox[key] = value

            # 2. Extract Clean Text
            full_text = wikicode.strip_code()
            # Specific cleanup for this use case
            if "Appearances" in full_text:
                full_text = full_text.split("Appearances")[0]

            return {
                "content": full_text.strip(),
                "infobox": infobox
            }
        except Exception as e:
            logger.error(f"Error parsing wikitext: {e}")
            return {"content": "", "infobox": {}}


class FandomScraper:
    """Main class to orchestrate data download"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})

    def _make_request(self, params: Dict[str, Any]) -> Optional[Dict]:
        """Internal wrapper to handle network errors and retries"""
        try:
            response = self.session.get(
                self.config.base_url, 
                params=params, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}. Retrying in {self.config.retry_delay}s...")
            time.sleep(self.config.retry_delay)
            return None

    def get_total_articles(self) -> int:
        """Get the total article statistics"""
        params = {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "statistics",
            "format": "json"
        }
        data = self._make_request(params)
        if data:
            return data["query"]["statistics"]["articles"]
        return 0

    def get_all_pages_generator(self) -> Generator[Dict, None, None]:
        """Generator that iterates over all pages (A-Z) handling pagination"""
        ap_from = ""
        
        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": "500",
                "apnamespace": "0",
                "apfilterredir": "nonredirects",
                "format": "json"
            }
            if ap_from:
                params["apfrom"] = ap_from
            
            data = self._make_request(params)
            if not data:
                break
            
            if "query" in data and "allpages" in data["query"]:
                pages = data["query"]["allpages"]
                for page in pages:
                    yield page
                
                if "continue" in data:
                    ap_from = data["continue"]["apcontinue"]
                else:
                    break
            else:
                break

    def fetch_page_detail(self, page_id: int, title: str) -> Optional[Dict[str, Any]]:
        """Download and process an individual page"""
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "pageids": str(page_id),
            "format": "json"
        }
        
        data = self._make_request(params)
        if not data:
            return None

        try:
            # Navigate the MediaWiki response JSON
            raw_text = data["query"]["pages"][str(page_id)]["revisions"][0]["*"]
            
            # Use our static Parser class
            parsed_data = WikiParser.clean_text(raw_text)
            
            return {
                "id": page_id,
                "title": title,
                "url": f"https://gameofthrones.fandom.com/wiki/?curid={page_id}",
                **parsed_data  # Merge of clean dictionary (content + infobox)
            }
        except KeyError:
            logger.error(f"Structure error in page {page_id}")
            return None

    def run(self, output_path: str):
        """
        Runs the process. If the file already exists, it resumes from where it left off
        and avoids duplicates.
        """
        total_remote = self.get_total_articles()
        
        # 1. Load already processed IDs to avoid repeating
        existing_ids = set()
        if os.path.exists(output_path):
            logger.info(f"📂 Found existing file at {output_path}. Checking processed IDs...")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing_ids.add(record['id'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"✅ Resuming... {len(existing_ids)} articles already downloaded.")
        else:
            logger.info(f"🚀 Starting fresh scrape of {total_remote} articles.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 2. Use 'a' mode (append) instead of 'w'
        with open(output_path, 'a', encoding='utf-8') as f:
            # Progress bar
            progress_bar = tqdm(self.get_all_pages_generator(), total=total_remote, unit="page")
            
            skipped_count = 0
            
            for page_meta in progress_bar:
                page_id = page_meta['pageid']
                title = page_meta['title']
                
                # 3. CHECK: If we already have it, skip
                if page_id in existing_ids:
                    skipped_count += 1
                    # Update the progress bar description for visual feedback
                    progress_bar.set_description(f"Skipping {title[:15]}...")
                    continue
                
                # If it doesn't exist, download it
                progress_bar.set_description(f"Downloading {title[:15]}...")
                content = self.fetch_page_detail(page_id, title)
                
                if content:
                    f.write(json.dumps(content) + "\n")
                    # Good practice to flush periodically or rely on OS buffer
                    # f.flush() 
                
                time.sleep(0.1)
                
        logger.info(f"🎉 Scraping finished. Skipped {skipped_count} existing articles.")