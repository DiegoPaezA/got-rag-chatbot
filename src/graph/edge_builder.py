import json
import os
import re
import logging
from tqdm import tqdm
from typing import Dict, List, Set, Any, Tuple

logger = logging.getLogger(__name__)
from src.utils.text import clean_value, SPLIT_PATTERN


class EdgeBuilder:
    """Build relationship edges from node properties with schema validation."""

    def __init__(self, data_dir: str):
        """Initialize paths and relationship/schema mappings.

        Args:
            data_dir: Directory containing node files (cleaned, validated, or raw).
        """
        self.data_dir = data_dir

        # Prioritize cleaned > validated > raw nodes
        self.nodes_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        if not os.path.exists(self.nodes_path):
            logger.warning(f"⚠️ nodes_cleaned.jsonl not found. Trying nodes_validated.jsonl")
            self.nodes_path = os.path.join(data_dir, "nodes_validated.jsonl")

        if not os.path.exists(self.nodes_path):
            logger.warning(f"⚠️ nodes_validated.jsonl not found. Using raw nodes.jsonl")
            self.nodes_path = os.path.join(data_dir, "nodes.jsonl")

        self.edges_path = os.path.join(data_dir, "edges.jsonl")
        self.alias_map: Dict[str, Set[str]] = {}

        # Inverse relationship mapping (e.g., CHILD_OF <-> PARENT_OF)
        self.inverse_map = {
            "CHILD_OF": "PARENT_OF", "PARENT_OF": "CHILD_OF",
            "SUCCEEDED_BY": "PRECEDED_BY", "PRECEDED_BY": "SUCCEEDED_BY",
            "OWNS_WEAPON": "OWNED_BY", "OWNED_BY": "OWNS_WEAPON",
            "VASSAL_OF": "OVERLORD_OF", "SWORN_TO": "HAS_MEMBER",
        }

        # Schema constraints: relation_type -> [allowed_source_node_types]
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
            "FOLLOWS_RELIGION": ["Character", "Organization", "House"],
            
            # War
            "PARTICIPANT_IN": ["Character", "House", "Organization", "Creature"],
            "COMMANDED_BY":   ["Battle", "Army"],
            "PART_OF_CONFLICT":["Battle"],
            "PART_OF_WAR":    ["Battle"],
            
            # Objects
            "CREATED_BY":     ["Object"],
            "OWNED_BY":       ["Object", "Creature"],
            "WIELDED_BY":     ["Object"],
            "OWNS_WEAPON":    ["Character", "House"],
            "HAS_ARMS":       ["Character", "House"],
            
            # Meta / Production
            "PLAYED_BY":      ["Character"],
            "DIED_IN_EPISODE":["Character", "Creature"],
            "APPEARED_IN_SEASON": ["Character", "Creature"]
        }

        # Property-to-relationship mapping (e.g., "Father" -> CHILD_OF)
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

    def _create_edge_and_inverse(self, source_id: str, rel_type: str, target_id: str, edges_set: Set[Tuple[str, str, str]], stats: Dict) -> None:
        """Create direct edge and its inverse/symmetric counterpart if applicable.

        Args:
            source_id: Source node ID.
            rel_type: Relationship type.
            target_id: Target node ID.
            edges_set: Set to add edge tuples to (for automatic deduplication).
            stats: Statistics dict to increment counters.
        """
        
        # Add direct edge
        direct_edge = (source_id, rel_type, target_id)
        if direct_edge not in edges_set:
            edges_set.add(direct_edge)
            stats["created"] += 1

        # Add inverse or symmetric edge if defined
        inverse_rel_type = self.inverse_map.get(rel_type)
        
        if inverse_rel_type:
            # Inverse: relationship type and direction change
            inverse_edge = (target_id, inverse_rel_type, source_id)
            if inverse_edge not in edges_set:
                edges_set.add(inverse_edge)
                stats["inverse_created"] += 1
                
        elif rel_type in ["MARRIED_TO", "SIBLING_OF", "LOVER_OF"]:
            # Symmetric: relationship is the same in both directions
            symmetric_edge = (target_id, rel_type, source_id)
            if symmetric_edge not in edges_set:
                edges_set.add(symmetric_edge)
                stats["inverse_created"] += 1

    def run(self) -> None:
        """Generate edges from nodes and validate against schema constraints.
        
        Estrategia:
        - Prioriza `normalized_relations` (salida del LLM).
        - Hace fallback a `properties` solo para campos relacionales que NO estén ya limpios.
        - Usa `edges_set` como set de tuplas (source, relation, target) para evitar duplicados.
        """

        if not os.path.exists(self.nodes_path):
            logger.error(f"❌ Nodes file not found at {self.nodes_path}")
            return

        logger.info(f"🔗 Building Edges from {self.nodes_path}...")
        
        nodes_map = self._load_nodes()
        self.alias_map = self._build_alias_map(nodes_map)

        edges_set: Set[Tuple[str, str, str]] = set()
        stats: Dict[str, int] = {"created": 0, "skipped_schema": 0, "inverse_created": 0}

        for node_id, node in tqdm(nodes_map.items(), desc="Processing Nodes"):
            node_type = node.get("type", "Lore")

            raw_props: Dict[str, Any] = node.get("properties", {}) or {}
            norm_rel: Dict[str, Any] = node.get("normalized_relations", {}) or {}

            # Defensive merge: prioritize LLM-normalized relations over raw properties
            #    Start with what the LLM already cleaned
            rel_source: Dict[str, Any] = dict(norm_rel)

            for key, val in raw_props.items():
                clean_key = key.strip()

                # AKA only used for aliases, not for edges
                if clean_key == "AKA":
                    continue

                # If already cleaned version of this key exists, skip (LLM wins)
                if clean_key in rel_source:
                    continue

                # Only consider properties that map to relationships
                if clean_key in self.rel_map:
                    # Clean key: remove infobox metadata like (children)
                    if "(" in clean_key:
                        clean_key = clean_key.split("(")[0].strip()

                    # Recheck with cleaned key
                    if clean_key not in rel_source:
                        rel_source[clean_key] = val

            # Process all merged relationship fields
            for clean_key, val in rel_source.items():
                rel_type = self.rel_map.get(clean_key)
                if not rel_type:
                    continue

                # Schema validation
                allowed_sources = self.schema_constraints.get(rel_type)
                if allowed_sources and node_type not in allowed_sources:
                    stats["skipped_schema"] += 1
                    continue

                # Parse targets based on source type
                if isinstance(val, list):
                    # LLM output: list already clean
                    targets = val
                else:
                    # Raw infobox string: parse and clean targets
                    val_str = str(val).replace(" and ", ",")
                    parts = [t.strip() for t in re.split(SPLIT_PATTERN, val_str) if t.strip()]
                    targets = [clean_value(t) for t in parts]

                for target in targets:
                    target_resolved = self._resolve_target(target)

                    # Extra heuristic for BELONGS_TO: auto-prefix "House" for characters
                    if (rel_type == "BELONGS_TO" and node_type == "Character" and target_resolved not in nodes_map):
                        candidate = f"House {target_resolved}"
                        if candidate in nodes_map and nodes_map[candidate].get("type") == "House":
                            target_resolved = candidate

                    if self._is_valid_target(target_resolved, node_id):
                        self._create_edge_and_inverse(
                            source_id=node_id,
                            rel_type=rel_type,
                            target_id=target_resolved,
                            edges_set=edges_set,
                            stats=stats,
                        )

            # 3) Edges de temporada para personajes
            if node_type == "Character":
                self._add_season_edges(node, edges_set)

        # 4) Convertir set de tuplas a lista de diccionarios
        edges_list = [
            {"source": s, "relation": r, "target": t}
            for (s, r, t) in edges_set
        ]

        self._save_edges(edges_list)
        logger.info("📊 Edge Creation Statistics:")
        logger.info(f"   Total Unique Edges Created: {len(edges_list)}")
        logger.info(f"   Relationships Skipped by Schema: {stats['skipped_schema']}")
        logger.info(f"   Inferred/Symmetric Edges Created: {stats['inverse_created']}")
        logger.info(f"✅ Edge building finished. Output: {self.edges_path}")

    def _load_nodes(self) -> Dict[str, Dict]:
        """Load nodes from JSONL file into a dictionary indexed by ID."""
        nodes: Dict[str, Dict] = {}
        with open(self.nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    n = json.loads(line)
                    nid = n.get("id")
                    if not nid:
                        continue
                    nodes[nid] = n
                except Exception:
                    continue
        return nodes

    def _build_alias_map(self, nodes_map: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """Build canonical ID mapping from aliases (AKA, titles, surnames).
        
        Creates normalized alias lookup table to resolve entity references.
        Only includes unambiguous mappings (1 alias -> 1 node ID).
        
        Args:
            nodes_map: Dictionary of all nodes indexed by ID.
            
        Returns:
            Dictionary mapping normalized aliases to their canonical node IDs.
        """
        alias_map: Dict[str, Set[str]] = {}

        def add_alias(alias: str, nid: str) -> None:
            """Add normalized alias to mapping.
            
            Args:
                alias: Alias string to add.
                nid: Node ID to map alias to.
            """
            alias = alias.strip()
            if not alias:
                return
            key = self._norm_alias(alias)
            if not key:
                return
            if key not in alias_map:
                alias_map[key] = set()
            alias_map[key].add(nid)

        for nid, node in nodes_map.items():
            props = node.get("properties", {})
            norm_rel = node.get("normalized_relations", {})

            add_alias(nid, nid)

            # Titles and AKA (prioritize normalized_relations)
            if props.get("Title"):
                add_alias(props["Title"], nid)

            aka_list: List[str] = []
            if isinstance(norm_rel.get("AKA"), list):
                aka_list = norm_rel["AKA"]
            elif "AKA" in props:
                aka_raw = props["AKA"]
                aka_list = [t.strip() for t in re.split(SPLIT_PATTERN, aka_raw) if t.strip()]

            for aka in aka_list:
                add_alias(aka, nid)

            # For houses, map surname to full name (e.g., "Stark" -> "House Stark")
            if node.get("type") == "House":
                if nid.lower().startswith("house "):
                    surname = nid[6:].strip()
                    if surname:
                        add_alias(surname, nid)
                if props.get("Type"):
                    for token in re.split(SPLIT_PATTERN, props["Type"]):
                        tok = token.strip()
                        if tok:
                            add_alias(tok, nid)

        cleaned_alias_map: Dict[str, Set[str]] = {}
        for key, ids in alias_map.items():
            if len(ids) == 1:
                cleaned_alias_map[key] = ids

        logger.info(f"🧭 Alias map built with {len(cleaned_alias_map)} unique aliases.")
        return cleaned_alias_map

    def _norm_alias(self, s: str) -> str:
        """Normalize alias to lowercase alphanumeric for matching.
        
        Args:
            s: String to normalize.
            
        Returns:
            Lowercase alphanumeric string with all non-alphanumeric chars removed.
        """
        return "".join(ch.lower() for ch in s if ch.isalnum())

    def _resolve_target(self, raw_target: str) -> str:
        """Resolve a target name to its canonical node ID via alias map.
        
        Args:
            raw_target: Raw target name from relationship property.
            
        Returns:
            Canonical node ID if found in alias map, otherwise raw_target unchanged.
        """
        key = self._norm_alias(raw_target)
        if not key:
            return raw_target

        ids = self.alias_map.get(key)
        if not ids or len(ids) != 1:
            return raw_target

        return next(iter(ids))

    def _add_season_edges(self, node: Dict, edges_set: Set[Tuple[str, str, str]]) -> None:
        """Add season appearance edges for character nodes.
        
        Extracts season numbers from Season or Appearances properties and creates
        APPEARED_IN_SEASON edges.
        
        Args:
            node: Character node dictionary.
            edges_set: Set to add season edges to.
        """
        props = node.get("properties", {})
        season_val = props.get("Season") or props.get("Appearances")

        if season_val:
            for s_num in re.findall(r"\d+", str(season_val)):
                if len(s_num) in {1, 2}:
                    edges_set.add((node['id'], "APPEARED_IN_SEASON", f"Season {s_num}"))

    def _is_valid_target(self, target: str, source_id: str) -> bool:
        """Check if target is valid for edge creation.
        
        Args:
            target: Target node ID.
            source_id: Source node ID.
            
        Returns:
            True if target is non-empty, distinct from source, not a note, and has sufficient length.
        """
        if not target:
            return False
        if target == source_id:
            return False
        if len(target) < 3:
            return False
        if "note" in target.lower():
            return False
        return True

    def _save_edges(self, edges: List[Dict]) -> None:
        """Write edges to JSONL file.
        
        Args:
            edges: List of edge dictionaries with 'source', 'relation', and 'target' keys.
        """
        with open(self.edges_path, "w", encoding="utf-8") as f:
            for e in edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")