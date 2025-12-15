import json
import os
import logging
from typing import Tuple, Dict, Any, Optional
from src.utils.text import clean_value, normalize_infobox_value, SKIP_KEYS

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Extract nodes and documents from raw wiki JSON dump.
    
    Performs heuristic classification of entities into types based on infobox fields
    and content analysis.
    """
    
    def __init__(self, input_path: str, output_dir: str) -> None:
        """Initialize builder with input and output paths.
        
        Args:
            input_path: Path to wiki dump JSONL file.
            output_dir: Output directory for nodes and documents files.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.nodes_file = os.path.join(output_dir, "nodes.jsonl")
        self.docs_file = os.path.join(output_dir, "documents.jsonl")

    def build(self) -> None:
        """Extract heuristic nodes and documents from wiki dump.
        
        Process flow:
        1. Type classification based on infobox fields and content
        2. Property cleaning and normalization
        3. Node creation with type scores for validation
        4. Document creation for RAG retrieval
        5. Save to nodes.jsonl and documents.jsonl
        """
        if not os.path.exists(self.input_path):
            logger.error(f"Input file not found: {self.input_path}")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        nodes = []
        documents = []
        stats = {"processed": 0, "types": {}}
        
        logger.info(f"🏗️  Extracting Nodes from {self.input_path}...")
        
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                title = data.get("title")
                if not title: continue
                
                entity_id = clean_value(str(title))
                infobox = data.get("infobox") or {}
                content = data.get("content") or ""

                # Advanced type classification (heuristic scoring)
                e_type, conf, reason, scores = self._resolve_type(entity_id, infobox, content)

                # Update statistics
                stats["types"][e_type] = stats["types"].get(e_type, 0) + 1

                # Clean infobox properties
                clean_props = {}
                for k, v in infobox.items():
                    if k is None: continue
                    if str(k).lower() in SKIP_KEYS: continue
                    
                    v_clean = clean_value(normalize_infobox_value(v))
                    if v_clean:
                        clean_props[k] = v_clean

                # Create node with cleaned properties
                # Store 'properties' for later EdgeBuilder processing
                nodes.append({
                    "id": entity_id,
                    "type": e_type,
                    "confidence": conf,
                    "reason": reason,
                    "type_scores": scores,  # Critical for validator
                    "properties": clean_props,
                    "url": data.get("url", "")
                })
                
                # Create document for RAG retrieval
                if len(content) > 50:
                    documents.append({
                        "id": entity_id,
                        "text": f"{entity_id} is a {e_type}. {content}",
                        "metadata": {
                            "type": e_type,
                            "source": "wiki_dump",
                            "confidence": conf
                        }
                    })
                
                stats["processed"] += 1

        self._save_jsonl(nodes, self.nodes_file)
        self._save_jsonl(documents, self.docs_file)
        
        logger.info(f"✅ Heuristic Build Done. Processed: {stats['processed']}")
        logger.info(f"   Distribution: {json.dumps(stats['types'], indent=2)}")

    def _save_jsonl(self, data: list, path: str) -> None:
        """Write data to JSONL file.
        
        Args:
            data: List of dictionaries to serialize.
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # =========================================================================
    # Type Scoring Logic (Entity Classification)
    # =========================================================================

    def _get_type_scores(self, title: str, infobox: dict) -> Dict[str, int]:
        """Calculate type scores based on infobox fields and title.
        
        Args:
            title: Entity title.
            infobox: Infobox properties dictionary.
            
        Returns:
            Dictionary mapping entity types to confidence scores.
        """
        scores = {
            "Character": 0, "House": 0, "Organization": 0, "Battle": 0,
            "Location": 0, "Object": 0, "Creature": 0, "Religion": 0, "Episode": 0
        }

        keys = set(k.lower().strip() for k in infobox.keys())
        type_raw = infobox.get("Type", "")
        if isinstance(type_raw, list):
            type_raw = " ".join(str(x) for x in type_raw)
        type_val = str(type_raw).lower()
        title_lower = title.lower()

        # --- CHARACTER ---
        if "actor" in keys or "played by" in keys: scores["Character"] += 5
        if "born" in keys or "died" in keys: scores["Character"] += 4
        if "father" in keys or "mother" in keys: scores["Character"] += 3
        if "lovers" in keys or "spouse" in keys: scores["Character"] += 3
        if any(role in type_val for role in ["king", "queen", "lord", "lady", "prince", "princess"]):
            scores["Character"] += 2
        famous_houses = ["stark", "targaryen", "lannister", "baratheon", "tully", "arryn", "greyjoy", "martell", "tyrell"]
        if any(name in title_lower for name in famous_houses): scores["Character"] += 1

        # --- HOUSE ---
        if title_lower.startswith("house "): scores["House"] += 10
        if "noble house" in type_val or "great house" in type_val: scores["House"] += 5
        if "seat" in keys: scores["House"] += 4
        if "sigil" in keys or "words" in keys: scores["House"] += 3
        if "overlords" in keys: scores["House"] += 3

        # --- EPISODE ---
        if "director" in keys or "writer" in keys: scores["Episode"] += 5
        if "air date" in keys or "runtime" in keys: scores["Episode"] += 4
        if "season" in keys and "episode" in keys: scores["Episode"] += 4
        if "episode" in title_lower: scores["Episode"] += 2

        # --- BATTLE ---
        if "battle" in title_lower or "siege" in title_lower or "sack of" in title_lower: scores["Battle"] += 6
        if "combatants" in keys: scores["Battle"] += 5
        if "commanders" in keys: scores["Battle"] += 4
        if "casualties" in keys or "conflict" in keys or "war" in keys: scores["Battle"] += 3

        # --- ORGANIZATION ---
        org_keywords = ["order", "guild", "bank", "company", "brotherhood", "guard", "watch", "council", "guildhall", "maesters"]
        if any(kw in type_val for kw in org_keywords) or any(kw in title_lower for kw in org_keywords):
            scores["Organization"] += 4
        org_keys = ["leader", "grand master", "lord commander", "headquarters", "members", "membership", "founded", "dissolved"]
        if any(k in keys for k in org_keys): scores["Organization"] += 3
        if title_lower in ["night's watch", "nights watch", "kingsguard", "iron bank"]: scores["Organization"] += 5
        if "actor" in keys: scores["Organization"] -= 5

        # --- CREATURE ---
        creature_keywords = ["dragon", "direwolf", "wolf", "horse", "bear", "raven"]
        if any(kw in type_val for kw in creature_keywords) or any(kw in title_lower for kw in creature_keywords):
            scores["Creature"] += 5
        if "species" in keys: scores["Creature"] += 4

        # --- OBJECT ---
        object_keywords = ["sword", "blade", "weapon", "crown", "ship", "poison", "armor", "shield"]
        if any(kw in type_val for kw in object_keywords) or any(kw in title_lower for kw in object_keywords):
            scores["Object"] += 4
        if "creator" in keys or "material" in keys: scores["Object"] += 3
        if "owners" in keys or "wielder" in keys: scores["Object"] += 3
        if "father" in keys or "mother" in keys: scores["Object"] -= 5

        # --- LOCATION ---
        loc_keywords = ["city", "castle", "region", "island", "river", "forest", "town", "village", "harbor", "bay", "sea", "mountain", "valley", "fortress"]
        if any(kw in type_val for kw in loc_keywords) or any(kw in title_lower for kw in loc_keywords):
            scores["Location"] += 3
        if "region" in keys or "geography" in keys: scores["Location"] += 4
        if "rulers" in keys or "lord" in keys: scores["Location"] += 2

        # --- RELIGION ---
        if "religion" in type_val or "god" in type_val or "faith" in type_val: scores["Religion"] += 5
        if "holy text" in keys or "deities" in keys: scores["Religion"] += 3

        return scores

    def _analyze_text_content(self, content: str) -> Optional[str]:
        """Fallback type inference from text content.
        
        Looks at the first 300 chars of content for type keywords.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Inferred entity type, or None if no match found.
        """
        if not content: return None
        intro = content[:300].lower()
        patterns = [
            ("noble house", "House"), ("great house", "House"),
            ("is a character", "Character"), ("was a character", "Character"),
            ("is a castle", "Location"), ("is a city", "Location"), ("is a region", "Location"),
            ("is a battle", "Battle"), ("was a battle", "Battle"),
            ("is a sword", "Object"), ("is a weapon", "Object"),
            ("is an episode", "Episode"), ("television episode", "Episode"),
        ]
        for pattern, t in patterns:
            if pattern in intro: return t
        return None

    def _resolve_type(self, title: str, infobox: dict, content: str) -> Tuple[str, str, str, dict]:
        """Resolve entity type using scores and content analysis.
        
        Decision logic:
        - Score >= 8: High confidence type classification
        - Score 4-7: Medium confidence, type is likely correct
        - Score 1-3: Low confidence, try text analysis for confirmation
        - Score <= 0: No signals, classify as generic Lore entity
        
        Args:
            title: Entity title.
            infobox: Infobox properties.
            content: Text content.
            
        Returns:
            Tuple of (entity_type, confidence, reason, scores_dict)
        """
        scores = self._get_type_scores(title, infobox)
        best_type, max_score = max(scores.items(), key=lambda x: x[1])

        # No signals in any field
        if max_score <= 0:
            text_guess = self._analyze_text_content(content)
            if text_guess:
                return text_guess, "Low", "Only inferred from text content", scores
            return "Lore", "Low", "No signals in scores or text", scores

        # Strong signals
        if max_score >= 8:
            return best_type, "High", f"Score: {max_score}", scores

        # Moderate signals
        if max_score >= 4:
            return best_type, "Medium", f"Score: {max_score}", scores

        # Weak signals (1-3): try text analysis
        text_guess = self._analyze_text_content(content)
        if text_guess and text_guess != best_type:
            return best_type, "Low", f"Score: {max_score}, text suggests {text_guess}", scores

        return best_type, "Low", f"Score: {max_score}", scores