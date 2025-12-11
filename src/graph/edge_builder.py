import json
import os
import re
import logging
from typing import Dict, List, Set
from src.utils.text import clean_value, SPLIT_PATTERN

logger = logging.getLogger(__name__)

class EdgeBuilder:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Fallback logic: tries to use AI-validated nodes, if not, uses raw nodes
        self.nodes_path = os.path.join(data_dir, "nodes_validated.jsonl")
        if not os.path.exists(self.nodes_path):
            logger.warning(f"⚠️ Validated nodes not found at {self.nodes_path}. Using raw nodes.")
            self.nodes_path = os.path.join(data_dir, "nodes.jsonl")
            
        self.edges_path = os.path.join(data_dir, "edges.jsonl")

        # --- BUSINESS RULES (SCHEMA CONSTRAINTS) ---
        # Define what types of nodes (Source) can have what relationships.
        # This avoids errors like "A Sword is the father of Jon Snow".
        self.schema_constraints = {
            # Family and Personal
            "CHILD_OF":       ["Character", "Creature"],
            "PARENT_OF":      ["Character", "Creature"],
            "SIBLING_OF":     ["Character", "Creature"],
            "MARRIED_TO":     ["Character"],
            "LOVER_OF":       ["Character"],
            
            # Loyalty and Politics
            "BELONGS_TO":     ["Character", "Creature", "House"],
            "SWORN_TO":       ["House", "Character", "Organization"],
            "AFFILIATED_WITH":["Character", "Organization", "House"],
            "VASSAL_OF":      ["House"],
            "SUCCEEDED_BY":   ["Character", "House"],
            "PRECEDED_BY":    ["Character", "House"],
            "LED_BY":         ["Organization", "House", "Battle", "Army"],
            
            # Geography
            "LOCATED_IN":     ["Location", "Battle", "House", "City", "Castle", "Organization", "Event"],
            "SEATED_AT":      ["House"],
            
            # Culture and Religion
            "HAS_CULTURE":    ["Character", "House", "Location"],
            "FOLLOWS_RELIGION": ["Character", "Organization"],
            
            # War
            "PARTICIPANT_IN": ["Character", "House", "Organization", "Creature"],
            "COMMANDED_BY":   ["Battle", "Army"],
            "PART_OF_CONFLICT":["Battle"],
            "PART_OF_WAR":    ["Battle"],
            
            # Objects
            "CREATED_BY":     ["Object"],
            "OWNED_BY":       ["Object", "Creature"],
            "WIELDED_BY":     ["Object"],
            "OWNS_WEAPON":    ["Character"],
            "HAS_ARMS":       ["Character", "House"],
            
            # Meta / Production
            "PLAYED_BY":      ["Character"],
            "DIED_IN_EPISODE":["Character", "Creature"],
            "APPEARED_IN_SEASON": ["Character", "Creature"]
        }

        # --- COMPLETE MAPPING OF PROPERTIES TO RELATIONSHIPS ---
        self.rel_map = {
            "Father": "CHILD_OF", "Mother": "CHILD_OF",
            "Spouse": "MARRIED_TO", "Siblings": "SIBLING_OF", "Children": "PARENT_OF",
            "Lovers": "LOVER_OF", "House": "BELONGS_TO", "Allegiance": "SWORN_TO",
            "Affiliation": "AFFILIATED_WITH", "Overlords": "VASSAL_OF",
            "Successor": "SUCCEEDED_BY", "Predecessor": "PRECEDED_BY", "Leader": "LED_BY",
            "Region": "LOCATED_IN", "Seat": "SEATED_AT", "Culture": "HAS_CULTURE",
            "Religion": "FOLLOWS_RELIGION", "Combatants": "PARTICIPANT_IN",
            "Commanders": "COMMANDED_BY", "Conflict": "PART_OF_CONFLICT", "War": "PART_OF_WAR",
            "Creator": "CREATED_BY", "Owners": "OWNED_BY", "Wielder": "WIELDED_BY",
            "Weapon": "OWNS_WEAPON", "Ancestral Weapon": "OWNS_WEAPON", "Arms": "HAS_ARMS",
            "Actor": "PLAYED_BY", "DeathEp": "DIED_IN_EPISODE"
        }

    def run(self):
        """Generate edges from nodes and validate against schema constraints.
        
        Extracts relationships from node properties and validates that the source
        node type is allowed to have that relationship type. Tracks creation and
        schema violation statistics.
        """
        if not os.path.exists(self.nodes_path):
            logger.error(f"❌ Nodes file not found at {self.nodes_path}")
            return

        logger.info(f"🔗 Building Edges from {self.nodes_path}...")
        
        nodes_map = self._load_nodes()
        edges = []
        stats = {"created": 0, "skipped_schema": 0}

        for node_id, node in nodes_map.items():
            node_type = node.get("type", "Lore")
            props = node.get("properties", {})

            # --- A. Relationships based on Properties (Infobox) ---
            for key, val in props.items():
                clean_key = key.strip()
                # Extra cleanup for composite keys
                if "(" in clean_key:
                    clean_key = clean_key.split("(")[0].strip()
                
                if clean_key in self.rel_map:
                    rel_type = self.rel_map[clean_key]
                    
                    # 1. SCHEMA VALIDATION (Schema Check)
                    allowed_sources = self.schema_constraints.get(rel_type)
                    if allowed_sources and node_type not in allowed_sources:
                        # Example: If an 'Object' has 'Father', it's ignored.
                        stats["skipped_schema"] += 1
                        continue 

                    # 2. Pre-cleaning of value (handle 'and')
                    val_preclean = val.replace(" and ", ",")

                    # 3. Target generation
                    targets = [t.strip() for t in re.split(SPLIT_PATTERN, val_preclean) if t.strip()]
                    
                    for target in targets:
                        target_clean = clean_value(target)
                        if self._is_valid_target(target_clean, node_id):
                            edges.append({
                                "source": node_id,
                                "relation": rel_type,
                                "target": target_clean
                            })

            # --- B. Special Logic (Seasons) ---
            if node_type == "Character":
                self._add_season_edges(node, edges)

        # Deduplicate (convert to hashable tuples and back to dict)
        unique_edges = [dict(t) for t in {tuple(d.items()) for d in edges}]
        
        self._save_edges(unique_edges)
        logger.info(f"✅ Edges built: {len(unique_edges)}. Skipped by Schema: {stats['skipped_schema']}")

    def _load_nodes(self) -> Dict:
        """Load nodes from JSONL file into memory dictionary indexed by node ID.
        
        Returns:
            Dict: Mapping of node IDs to node objects.
        """
        nodes = {}
        with open(self.nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    n = json.loads(line)
                    nodes[n['id']] = n
                except: continue
        return nodes

    def _add_season_edges(self, node, edges_list):
        """Extract seasons from 'Season' or 'Appearances' properties and create edges.
        
        Searches for season numbers in properties and creates APPEARED_IN_SEASON edges.
        Filters out year numbers (e.g., '298 AC') using heuristic on digit count.
        """
        props = node.get("properties", {})
        season_val = props.get("Season") or props.get("Appearances")
        
        if season_val:
            # Search for digits: "Season 1, 2" -> ['1', '2']
            for s_num in re.findall(r"\d+", str(season_val)):
                # Heuristic filter: Seasons are 1 or 2 digits (avoid years like '298 AC')
                if len(s_num) in [1, 2]:
                    edges_list.append({
                        "source": node['id'],
                        "relation": "APPEARED_IN_SEASON",
                        "target": f"Season {s_num}"
                    })

    def _is_valid_target(self, target, source_id):
        """Validate target node quality for edge creation.
        
        Applies quality filters: non-empty, no self-loops, minimum length,
        and excludes footnote references.
        
        Args:
            target: Target node identifier string
            source_id: Source node identifier to check for self-loops
            
        Returns:
            bool: True if target passes all quality checks
        """
        if not target: return False
        if target == source_id: return False  # No self-loops
        if len(target) < 3: return False  # Avoid short noise
        if "note" in target.lower(): return False  # Avoid footnotes
        return True

    def _save_edges(self, edges):
        """Save deduplicated edges to JSONL file.
        
        Args:
            edges: List of edge dictionaries with source, relation, and target.
        """
        with open(self.edges_path, "w", encoding="utf-8") as f:
            for e in edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")