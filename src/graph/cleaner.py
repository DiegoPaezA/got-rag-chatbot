import os
import json
import time
import random
import logging
from typing import Dict, Any, List, Set

from dotenv import load_dotenv
from tqdm import tqdm

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Output Schemas for Structured LLM Responses
# =============================================================================

class SingleNodeCleaning(BaseModel):
    """Cleaned relations for a single node."""
    id: str = Field(description="The node id (exactly as in the graph).")
    normalized_relations: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map from property name (e.g. Father, House, AKA) to a list of cleaned entity names."
    )


class BatchCleaningResult(BaseModel):
    """LLM response for a batch of nodes."""
    results: List[SingleNodeCleaning]


# =============================================================================
# Main GraphCleaner Class
# =============================================================================

class GraphCleaner:
    """
    Use an LLM to normalize relationship-bearing properties in nodes_validated.jsonl,
    producing a new file nodes_cleaned.jsonl with a `normalized_relations` field per node.

    - Input:  nodes_validated.jsonl (fallback: nodes.jsonl)
    - Output: nodes_cleaned.jsonl
    - Checkpoint: nodes_cleaner_checkpoint.jsonl
    """

    def __init__(self, data_dir: str, config_path: str = "cfg/config.json"):
        self.data_dir = data_dir

        # Prefer validated nodes; fall back to raw nodes if absent.
        self.nodes_in_path = os.path.join(data_dir, "nodes_validated.jsonl")
        if not os.path.exists(self.nodes_in_path):
            logger.warning("⚠️ nodes_validated.jsonl not found. Falling back to nodes.jsonl")
            self.nodes_in_path = os.path.join(data_dir, "nodes.jsonl")

        self.nodes_out_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        self.checkpoint_path = os.path.join(data_dir, "nodes_cleaner_checkpoint.jsonl")

        self.batch_size = 10
        self.max_retries = 5
        self.base_delay = 4.0

        self.config = self._load_config(config_path)
        self.llm_settings = self.config.get("llm_settings", {})
        self.prompts = self.config.get("prompts", {})

        # Properties to normalize (includes AKA)
        self.target_keys: Set[str] = {
            "Father", "Mother", "Children", "Issue", "Siblings",
            "Spouse", "Lovers",
            "House", "Affiliation", "Allegiance", "Overlords", "Vassals",
            "Owners", "Creator", "Weapon", "Ancestral Weapon",
            "Culture", "Religion",
            "Combatants", "Commanders", "Conflict", "War",
            "AKA",
        }

        self.chain = None  # LLM chain

    # =========================================================================
    # Configuration Loading and LLM Initialization
    # =========================================================================

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration file safely.
        
        Args:
            path: Path to configuration file.
            
        Returns:
            Configuration dictionary, or empty dict if file not found or on error.
        """
        if not os.path.exists(path):
            logger.warning(f"⚠️ Config file not found at {path}. Using defaults.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {}

    def _init_chain(self) -> None:
        """Initialize LLM model and prompt template for batch cleaning.
        
        Sets up Google Generative AI model with structured output schema
        and loads system/human prompts from config.
        """
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY not found in environment.")

        model_name = self.llm_settings.get("model_name", "gemini-2.5-flash")
        temperature = self.llm_settings.get("temperature", 0.0)
        max_retries_llm = self.llm_settings.get("max_retries", 5)

        logger.info(f"🧹 Initializing GraphCleaner LLM: {model_name} (T={temperature})")

        llm_raw = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=max_retries_llm,
        )

        llm = llm_raw.with_structured_output(BatchCleaningResult)

        # Default system and human prompts if not in config
        # These guide the LLM on relationship normalization rules

        default_system_prompt = """
You are a data cleaning assistant for a Game of Thrones knowledge graph.

You receive a BATCH of nodes. For each node you are given:
- id: the node identifier (e.g. "Jon Snow").
- type: the node type (Character, House, Object, etc.).
- raw_properties: the RAW infobox properties extracted from a fan wiki (possibly messy).

Your goal is to build a CLEAN view of relationship-bearing fields, by splitting
concatenated names into a list of individual entities.

Focus primarily on properties that refer to other entities, such as (if present):
- Father, Mother, Children, Issue, Siblings
- Spouse, Lovers
- House, Affiliation, Allegiance, Overlords, Vassals
- Owners, Creator, Weapon, Ancestral Weapon
- Culture, Religion
- Combatants, Commanders, Conflict, War
- AKA  (alternative names / aliases for the same entity)

Rules:
1. Use the raw property text AS GROUND TRUTH. Do not invent entities that are not hinted in the text.
2. Split concatenated names into separate items, for example:
   - "Rhaegar Targaryen Eddard Stark"  -> ["Rhaegar Targaryen", "Eddard Stark"]
   - "House StarkNights Watch"        -> ["House Stark", "Night's Watch"]
   - "Robb StarkJon Snow Sansa Stark" -> ["Robb Stark", "Jon Snow", "Sansa Stark"]
3. Preserve full names as they would appear in a wiki page title:
   examples: "Jon Snow", "Daenerys Targaryen", "House Stark", "Night's Watch".
4. If a string is ambiguous and you are not confident in splitting, keep it as a single item.
5. Ignore descriptive fields such as Birth, Death, Titles, Series, Season, Appearances, Actor.
6. If a property is missing or clearly empty, use an empty list for that property.
7. Do NOT rename entities arbitrarily; prefer the text as written in the raw properties.

Output format (very important):
- You MUST return a JSON object.
- The top-level object must have exactly one key named "results".
- "results" must be a list.
- Each element of "results" must be an object with:
  - "id": the same id you received for that node.
  - "normalized_relations": an object that maps property names (e.g. "Father", "House", "AKA")
    to a list of cleaned entity strings.

If for a given node you do not find any relationship-bearing properties,
use an empty object for "normalized_relations".

Do not include explanations or markdown in your answer. Return only the JSON object.
"""

        default_human_prompt = """
Here is a batch of nodes to clean.

NODES JSON (list of objects with id, type and raw_properties):
{nodes_json}

Return ONLY the JSON object with the described structure.
"""

        system_tmpl = self.prompts.get("cleaner_system", default_system_prompt)
        human_tmpl = self.prompts.get("cleaner_human", default_human_prompt)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_tmpl),
            ("human", human_tmpl),
        ])

        self.chain = prompt | llm

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self) -> None:
        """Main entrypoint: clean relationship fields and produce nodes_cleaned.jsonl.
        
        Process flow:
        1. Load all nodes from input file
        2. Check checkpoint for already-processed nodes
        3. Identify candidates needing cleaning
        4. Process in batches with LLM and retry on quota errors
        5. Consolidate results into final output file
        """
        if not os.path.exists(self.nodes_in_path):
            logger.error(f"❌ Nodes file not found: {self.nodes_in_path}")
            return

        self._init_chain()

        # 1. Load all nodes
        all_nodes = self._load_jsonl(self.nodes_in_path)
        logger.info(f"📦 Loaded {len(all_nodes)} nodes from {self.nodes_in_path}")

        # 2. Already processed IDs (checkpoint)
        processed_ids = self._load_checkpoint_ids()
        logger.info(f"🔁 Cleaner checkpoint has {len(processed_ids)} nodes")

        # 3. Build candidate list
        candidates: List[Dict[str, Any]] = []
        for node in all_nodes:
            nid = node.get("id")
            if not nid or nid in processed_ids:
                continue
            if self._should_clean(node):
                candidates.append(node)

        logger.info(f"🎯 Nodes to clean in this run: {len(candidates)}")

        # If no candidates, only consolidate
        if not candidates:
            self._consolidate_results(all_nodes)
            logger.info("✅ No new candidates. Consolidation done.")
            return

        # 4. Process batch by batch, append to checkpoint
        with open(self.checkpoint_path, "a", encoding="utf-8") as f_cp:
            iterator = tqdm(
                range(0, len(candidates), self.batch_size),
                desc="🧹 Cleaning nodes",
                unit="batch",
            )

            for i in iterator:
                batch = candidates[i : i + self.batch_size]
                cleaned_batch = self._process_batch_with_retry(batch)

                for node in cleaned_batch:
                    f_cp.write(json.dumps(node, ensure_ascii=False) + "\n")

                f_cp.flush()
                os.fsync(f_cp.fileno())
                time.sleep(1)  # friendly to the API

        # 5. Consolidate into nodes_cleaned.jsonl
        self._consolidate_results(all_nodes)
        logger.info(f"✅ Cleaning finished. Output: {self.nodes_out_path}")

    # =========================================================================
    # Batch LLM Processing with Retry Logic
    # =========================================================================

    def _process_batch_with_retry(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of nodes through LLM with exponential backoff retry.
        
        Handles API quota errors (429) with exponential backoff.
        Returns original nodes if cleaning fails after max retries.
        
        Args:
            nodes: Batch of node dictionaries to clean.
            
        Returns:
            Nodes with normalized_relations field added.
        """
        llm_nodes = []
        for node in nodes:
            raw_props = node.get("properties", {})
            llm_nodes.append({
                "id": node.get("id", ""),
                "type": node.get("type", ""),
                "raw_properties": raw_props,
            })

        payload = {
            "nodes_json": json.dumps(llm_nodes, ensure_ascii=False)
        }

        for attempt in range(self.max_retries):
            try:
                response: BatchCleaningResult = self.chain.invoke(payload)
                results_map: Dict[str, SingleNodeCleaning] = {
                    res.id: res for res in response.results
                }

                output_nodes: List[Dict[str, Any]] = []
                for node in nodes:
                    nid = node.get("id")
                    cleaned = results_map.get(nid)
                    if cleaned:
                        node["normalized_relations"] = cleaned.normalized_relations
                    output_nodes.append(node)
                return output_nodes

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (self.base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(
                        f"⚠️ Quota hit in GraphCleaner. Waiting {wait_time:.1f}s... "
                        f"(Attempt {attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Critical error in cleaning batch: {e}")
                    return nodes

        logger.error(f"❌ Failed cleaning batch after {self.max_retries} retries.")
        return nodes

    # =========================================================================
    # Candidate Node Selection
    # =========================================================================

    def _should_clean(self, node: Dict[str, Any]) -> bool:
        """Determine if a node should be cleaned by the LLM.
        
        Selects nodes of important types (Character, House, etc.) that have
        relationship-bearing properties but haven't been normalized yet.
        
        Args:
            node: Node dictionary to evaluate.
            
        Returns:
            True if node should be cleaned, False otherwise.
        """
        ntype = node.get("type", "")
        props = node.get("properties", {})

        # Only clean important node types
        if ntype not in ["Character", "House", "Organization", "Creature", "Object"]:
            return False

        # Skip if already cleaned
        if "normalized_relations" in node:
            return False

        # Clean if has at least one target relationship key
        for k in props.keys():
            if k in self.target_keys:
                return True

        return False

    # =========================================================================
    # File I/O Utilities
    # =========================================================================

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load all nodes from JSONL file.
        
        Args:
            path: Path to JSONL file.
            
        Returns:
            List of node dictionaries. Invalid lines are skipped.
        """
        data: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        continue
        return data

    def _load_checkpoint_ids(self) -> Set[str]:
        """Load IDs of nodes already processed from checkpoint file.
        
        Returns:
            Set of node IDs that have already been cleaned.
        """
        ids: Set[str] = set()
        if os.path.exists(self.checkpoint_path):
            logger.info(f"🔄 Found cleaner checkpoint: {self.checkpoint_path}")
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        n = json.loads(line)
                        nid = n.get("id")
                        if nid:
                            ids.add(nid)
                    except Exception:
                        continue
        return ids

    def _consolidate_results(self, original_nodes: List[Dict[str, Any]]) -> None:
        """Merge cleaned nodes from checkpoint with original nodes.
        
        Writes final consolidated output to nodes_cleaned.jsonl with all
        relationship fields normalized.
        
        Args:
            original_nodes: Original list of all nodes before cleaning.
        """
        corrections: Dict[str, Dict[str, Any]] = {}

        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        n = json.loads(line)
                        nid = n.get("id")
                        if nid:
                            corrections[nid] = n
                    except Exception:
                        continue

        final_nodes: List[Dict[str, Any]] = []
        for node in original_nodes:
            nid = node.get("id")
            if nid in corrections:
                final_nodes.append(corrections[nid])
            else:
                final_nodes.append(node)

        with open(self.nodes_out_path, "w", encoding="utf-8") as f:
            for n in final_nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")

        logger.info(
            f"💾 Consolidated cleaned nodes saved to {self.nodes_out_path} "
            f"(total: {len(final_nodes)})"
        )


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    cleaner = GraphCleaner(data_dir="data/processed")
    cleaner.run()
