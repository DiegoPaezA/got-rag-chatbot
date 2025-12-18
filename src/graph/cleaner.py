import os
import json
import time
import random
import logging
import hashlib
from typing import Dict, Any, List, Set, Optional

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
        description="Map from property name (e.g. Father, House, AKA, Titles) to a list of cleaned entity names."
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
    producing nodes_cleaned.jsonl with a `normalized_relations` field per node.
    """

    # Generic junk tokens that should never become entity targets
    GENERIC_BAD = {
        "unknown", "none", "n/a", "na", "?", "unnamed",
        "sons", "son", "son(s)", "daughters", "daughter",
        "children", "various", "numerous"
    }

    def __init__(self, data_dir: str, config_path: str = "cfg/config.json"):
        self.data_dir = data_dir

        # Prefer validated nodes; fall back to raw nodes if absent.
        self.nodes_in_path = os.path.join(data_dir, "nodes_validated.jsonl")
        if not os.path.exists(self.nodes_in_path):
            logger.warning("⚠️ nodes_validated.jsonl not found. Falling back to nodes.jsonl")
            self.nodes_in_path = os.path.join(data_dir, "nodes.jsonl")

        self.nodes_out_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        self.checkpoint_path = os.path.join(data_dir, "nodes_cleaner_checkpoint.jsonl")

        self.batch_size = 15
        self.max_retries = 5
        self.base_delay = 4.0

        self.config = self._load_config(config_path)
        self.llm_settings = self.config.get("llm_settings", {})
        self.prompts = self.config.get("prompts", {})

        # Properties to normalize
        self.target_keys: Set[str] = {
            # Family / couple
            "Father", "Mother", "Children", "Issue", "Siblings",
            "Spouse", "Lovers",

            # Politics / loyalties
            "House", "Affiliation", "Allegiance", "Overlords", "Vassals", 
            "Titles", # IMPORTANT: included here

            # Succession
            "Successor", "Predecessor", "Heir",

            # Leadership
            "Leader", "Head", "Rulers",

            # Geography
            "Region", "Seat",

            # Production / episodes
            "DeathEp", "Death",

            # War
            "Combatants", "Commanders", "Conflict", "War",

            # Objects
            "Owners", "Creator", "Weapon", "Ancestral Weapon",

            # “Problematic but useful to clean”
            "Culture", "Arms", "Actor",

            # AKA
            "AKA",
        }

        self.chain = None  # LLM chain

    # =========================================================================
    # Configuration Loading and LLM Initialization
    # =========================================================================

    def _load_config(self, path: str) -> Dict[str, Any]:
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

        # ---------------------------------------------------------------------
        # UPDATED PROMPT TO HANDLE CONCATENATED TITLES
        # ---------------------------------------------------------------------
        default_system_prompt = """
            You are a STRICT TEXT PROCESSING ENGINE for a dataset.
            You are NOT a knowledge base. You DO NOT know anything about Game of Thrones.

            INPUT: You receive a BATCH of nodes with 'raw_properties'.
            TASK: Clean, split, and normalize the STRINGS inside 'raw_properties'.
            
            CRITICAL ANTI-HALLUCINATION RULES:
            1. INPUT-OUTPUT PARITY: You may ONLY output property keys that exist in the input 'raw_properties'.
            2. NO EXTERNAL KNOWLEDGE: Do not infer facts. If the text says "Father: Unknown", output [].
            3. CONTENT ONLY: Process only the text provided.

            Focus on cleaning these properties (if they exist in input):
            - Family: Father, Mother, Children, Issue, Siblings
            - Couple: Spouse, Lovers
            - Politics: House, Affiliation, Titles (Look for concatenated titles!)
            - Succession: Successor, Predecessor, Heir
            - Production: DeathEp (episode title ONLY), Actor
            - AKA (alternative names)

            Processing Rules:

            1. SPLITTING & CONCATENATION (Critical for 'Titles' and 'Affiliation'):
               The data often misses spaces between words. You MUST split them based on capitalization.
               
               EXAMPLES:
               - "Lord of the Iron IslandsKing of Salt and Rock" -> ["Lord of the Iron Islands", "King of Salt and Rock"]
               - "Son of the Sea WindLord Reaper of Pyke" -> ["Son of the Sea Wind", "Lord Reaper of Pyke"]
               - "Euron Greyjoy Yara Greyjoy" -> ["Euron Greyjoy", "Yara Greyjoy"]
               - "House StarkNights Watch" -> ["House Stark", "Night's Watch"]
               - "998th Lord Commander of the Nights Watch King in the North" -> ["998th Lord Commander of the Night's Watch", "King in the North"]

            2. PARENTHETICAL REMOVAL:
               Remove status descriptions unless they are part of the name/title.
               - "Aegon (died young)" -> ["Aegon"]
               - "House Stark (formerly)" -> ["House Stark"]
               
            3. TITLES:
               - Preserve full titles if they appear distinct.
               - Separate concatenated titles into a LIST.

            4. GARBAGE REMOVAL: 
               If the text is generic ("sons", "unknown", "unnamed"), return [].

            Output format:
            - Return ONLY a valid JSON object with "results".
            """

        default_human_prompt = """
        Here is a batch of nodes to clean.

        NODES JSON:
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
    # Fingerprint / Incremental Re-clean
    # =========================================================================

    def _node_fingerprint(self, node: Dict[str, Any]) -> str:
        props = node.get("properties", {}) or {}
        payload = {
            "target_keys": sorted(list(self.target_keys)),
            "values": {k: props.get(k) for k in sorted(self.target_keys) if k in props},
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _load_checkpoint_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        if os.path.exists(self.checkpoint_path):
            logger.info(f"🔄 Found cleaner checkpoint: {self.checkpoint_path}")
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        n = json.loads(line)
                        nid = n.get("id")
                        fp = n.get("fp")
                        if nid and fp:
                            idx[nid] = fp
                    except Exception:
                        continue
        return idx

    # =========================================================================
    # Candidate Node Selection
    # =========================================================================

    def _should_clean(self, node: Dict[str, Any]) -> bool:
        ntype = node.get("type", "")
        props = node.get("properties", {}) or {}

        if ntype not in ["Character", "House", "Organization", "Creature", "Object"]:
            return False

        # Must have at least one relationship-bearing key with a non-empty value
        for k in self.target_keys:
            if props.get(k):
                return True
        return False

    # =========================================================================
    # Post-processing (Deterministic Filters)
    # =========================================================================

    def _postprocess_normalized_relations(self, rels: Dict[str, Any]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if not isinstance(rels, dict):
            return out

        for k, v in rels.items():
            if not isinstance(v, list):
                continue

            cleaned: List[str] = []
            for item in v:
                s = str(item).strip()
                if not s:
                    continue
                low = s.lower().strip()
                if low in self.GENERIC_BAD:
                    continue
                cleaned.append(s)

            if k == "DeathEp":
                cleaned = [x for x in cleaned if len(x) >= 3]
                if len(cleaned) > 1:
                    cleaned = [cleaned[0]]

            if k == "Actor" and len(cleaned) > 1:
                cleaned = [cleaned[0]]
            
            # Note: Do not limit 'Titles' to a single value; allow a list.

            out[k] = cleaned

        return out

    # =========================================================================
    # File I/O Utilities
    # =========================================================================

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
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

    def _consolidate_results(self, original_nodes: List[Dict[str, Any]]) -> None:
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

    # =========================================================================
    # Batch LLM Processing with Retry Logic
    # =========================================================================

    def _process_batch_with_retry(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        llm_nodes = []
        node_security_map: Dict[str, Set[str]] = {}

        for node in nodes:
            nid = node.get("id", "")
            raw_props_all = node.get("properties", {}) or {}
            
            raw_props = {k: raw_props_all.get(k) for k in self.target_keys if raw_props_all.get(k)}
            
            node_security_map[nid] = set(raw_props.keys())

            llm_nodes.append({
                "id": nid,
                "type": node.get("type", ""),
                "raw_properties": raw_props,
            })

        payload = {"nodes_json": json.dumps(llm_nodes, ensure_ascii=False)}

        for attempt in range(self.max_retries):
            try:
                response: BatchCleaningResult = self.chain.invoke(payload)
                results_map = {res.id: res for res in response.results}

                output_nodes = []
                for node in nodes:
                    nid = node.get("id")
                    cleaned = results_map.get(nid)
                    
                    if cleaned:
                        allowed_keys = node_security_map.get(nid, set())
                        safe_relations = {}
                        
                        for k, v in cleaned.normalized_relations.items():
                            if k in allowed_keys:
                                safe_relations[k] = v
                            else:
                                logger.warning(f"🛡️ Firewall: Blocked hallucinated property '{k}' for node '{nid}'")
                        
                        node["normalized_relations"] = self._postprocess_normalized_relations(safe_relations)
                    else:
                        node["normalized_relations"] = node.get("normalized_relations", {}) or {}

                    node["fp"] = self._node_fingerprint(node)
                    output_nodes.append(node)
                
                return output_nodes

            except Exception as e:
                logger.error(f"Error in batch: {e}")
                # Exponential backoff
                time.sleep(self.base_delay * (2 ** attempt))
        
        # Fallback if all retries fail
        return nodes

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self) -> None:
        if not os.path.exists(self.nodes_in_path):
            logger.error(f"❌ Nodes file not found: {self.nodes_in_path}")
            return

        self._init_chain()

        all_nodes = self._load_jsonl(self.nodes_in_path)
        logger.info(f"📦 Loaded {len(all_nodes)} nodes from {self.nodes_in_path}")

        processed = self._load_checkpoint_index()
        logger.info(f"🔁 Cleaner checkpoint index has {len(processed)} ids")

        candidates: List[Dict[str, Any]] = []
        for node in all_nodes:
            nid = node.get("id")
            if not nid:
                continue

            if not self._should_clean(node):
                continue

            fp = self._node_fingerprint(node)
            if nid in processed and processed[nid] == fp:
                continue

            candidates.append(node)

        logger.info(f"🎯 Nodes to clean in this run: {len(candidates)}")

        if not candidates:
            self._consolidate_results(all_nodes)
            logger.info("✅ No new candidates. Consolidation done.")
            return

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

        with open(self.checkpoint_path, "a", encoding="utf-8") as f_cp:
            iterator = tqdm(
                range(0, len(candidates), self.batch_size),
                desc="🧹 Cleaning nodes",
                unit="batch",
            )

            for i in iterator:
                batch = candidates[i: i + self.batch_size]
                cleaned_batch = self._process_batch_with_retry(batch)

                for node in cleaned_batch:
                    if "fp" not in node:
                        node["fp"] = self._node_fingerprint(node)
                    f_cp.write(json.dumps(node, ensure_ascii=False) + "\n")

                f_cp.flush()
                os.fsync(f_cp.fileno())
                time.sleep(1)

        self._consolidate_results(all_nodes)
        logger.info(f"✅ Cleaning finished. Output: {self.nodes_out_path}")


if __name__ == "__main__":
    cleaner = GraphCleaner(data_dir="data/processed")
    cleaner.run()