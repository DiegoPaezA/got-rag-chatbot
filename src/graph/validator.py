import json
import os
import time
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Set
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Output Schemas for Structured LLM Responses
# =============================================================================

class NodeValidation(BaseModel):
    """Validation result for a single node."""
    id: str = Field(description="The exact ID of the analyzed node.")
    suggested_type: str = Field(description="The correct type.")
    confidence: str = Field(description="'High', 'Medium', or 'Low'.")
    rationale: str = Field(description="Brief reason.")


class BatchValidationResult(BaseModel):
    """LLM response for a batch of nodes."""
    results: List[NodeValidation]

class GraphValidator:
    """Validate node types using LLM against knowledge base documents.
    
    Processes nodes in batches, validates them against document context,
    and produces nodes_validated.jsonl with corrected types.
    """
    
    def __init__(self, data_dir: str, config_path: str = "cfg/config.json") -> None:
        """Initialize validator with paths and configuration.
        
        Args:
            data_dir: Directory containing node and document files.
            config_path: Path to configuration file.
        """
        self.data_dir = data_dir
        
        # Input and output file paths
        self.nodes_path = os.path.join(data_dir, "nodes.jsonl")
        self.docs_path = os.path.join(data_dir, "documents.jsonl")
        self.checkpoint_path = os.path.join(data_dir, "nodes_llm_checkpoint.jsonl")
        self.final_path = os.path.join(data_dir, "nodes_validated.jsonl")
        
        # Load Configuration using ConfigManager
        config_manager = ConfigManager()
        validator_config = config_manager.get_processing_config("validator")
        
        self.batch_size = validator_config.get("batch_size", 10)
        self.max_text_length = validator_config.get("max_text_length", 600)
        self.min_text_length = validator_config.get("min_text_length", 20)
        
        self.llm_settings = config_manager.get("llm_settings", default={})
        self.prompts = config_manager.get("prompts", default={})
        self.allowed_types = config_manager.get("graph_settings", "allowed_types", default=[])

    def validate(self) -> None:
        """Main validation entrypoint: validate node types against knowledge base.
        
        Process flow:
        1. Initialize LLM model and prompts from config
        2. Load all nodes and document context
        3. Filter candidates needing validation
        4. Process batches through LLM with retry logic
        5. Consolidate results into validated output file
        """
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("❌ GOOGLE_API_KEY missing.")
            return

        model_name = self.llm_settings.get("model_name", "gemini-2.5-flash")
        logger.info(f"🤖 Initializing LLM validation ({model_name})")

        # Setup LLM and prompt templates from configuration
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=self.llm_settings.get("temperature", 0.0),
            max_retries=self.llm_settings.get("max_retries", 5),
        ).with_structured_output(BatchValidationResult)

        system_tmpl = self.prompts.get("validator_system", "")
        human_tmpl = self.prompts.get("validator_human", "{input_data}")
        
        self.chain = ChatPromptTemplate.from_messages([
            ("system", system_tmpl), ("human", human_tmpl)
        ]) | llm

        # Load nodes and track progress
        all_nodes = self._load_jsonl(self.nodes_path)
        processed_ids = self._load_checkpoint_ids()

        # Filter nodes needing validation
        candidates = []
        for node in all_nodes:
            if node['id'] in processed_ids: continue
            if self._should_validate(node):
                candidates.append(node)

        logger.info(f"📊 Total: {len(all_nodes)}. Processed: {len(processed_ids)}. To Validate: {len(candidates)}")

        if not candidates:
            self._consolidate_results(all_nodes)
            return

        # Load document contexts for candidates
        candidate_ids_set = {n['id'] for n in candidates}
        contexts_map = self._load_candidate_contexts(candidate_ids_set)

        # Process batches through LLM
        with open(self.checkpoint_path, "a", encoding="utf-8") as f_cp:
            process_iterator = tqdm(
                range(0, len(candidates), self.batch_size), 
                desc="🤖 Validating", 
                unit="batch"
            )

            for i in process_iterator:
                batch = candidates[i : i + self.batch_size]
                
                # Use retry logic here encapsulated
                validated_batch = self._process_batch_with_retry(batch, contexts_map)
                
                # Immediate save to checkpoint
                for node in validated_batch:
                    f_cp.write(json.dumps(node, ensure_ascii=False) + "\n")
                
                f_cp.flush()
                os.fsync(f_cp.fileno())                
                time.sleep(1)  # Rate limit friendly

        self._consolidate_results(all_nodes)

    # ==========================================================
    # BUSINESS LOGIC AND RETRY HANDLING
    # ==========================================================

    def _process_batch_with_retry(self, nodes: List[Dict], contexts: Dict[str, str]) -> List[Dict]:
        """Process a batch handling 429 errors explicitly."""
        
        # Prepare LLM input
        llm_input_data = []
        for node in nodes:
            nid = node['id']
            props_str = str(node.get("properties", {}))[:300]
            llm_input_data.append({
                "id": nid,
                "current_type": node.get("type"),
                "heuristic_confidence": node.get("confidence"),
                "properties": props_str,
                "text_context": contexts.get(nid, "No text available.")
            })

        max_retries = 5
        base_delay = 4

        for attempt in range(max_retries):
            try:
                # Invoke chain with dynamic parameters
                response = self.chain.invoke({
                    "allowed_types": ", ".join(self.allowed_types),
                    "input_data": json.dumps(llm_input_data, ensure_ascii=False)
                })
                
                # Map response to nodes
                results_map = {res.id: res for res in response.results}
                output = []
                for n in nodes:
                    res = results_map.get(n['id'])
                    if res:
                        # Update node with LLM validation
                        n["type"] = res.suggested_type
                        n["confidence"] = f"{res.confidence} (LLM)"
                        n["reason"] = res.rationale
                    output.append(n)
                return output

            except Exception as e:
                error_str = str(e)
                # Detect quota error (Gemini Free Tier)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(f"⚠️ Quota hit. Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Critical Batch Error: {e}")
                    return nodes  # Return unchanged to avoid breaking everything

        logger.error(f"❌ Failed batch after {max_retries} retries.")
        return nodes

    def _should_validate(self, node: Dict[str, Any]) -> bool:
        """Determine if a node should be validated by the LLM.
        
        Returns True if node has low confidence or ambiguous type scores.
        """
        # Basic criteria
        if node.get("confidence") == "Low": return True
        if node.get("type") in ["Lore", "Organization", "Object", "Creature"]: return True
        
        # Advanced criterion (close scores)
        scores = node.get("type_scores", {})
        if scores:
            sorted_vals = sorted(scores.values(), reverse=True)
            if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) <= 2:
                return True
        return False

    def _load_candidate_contexts(self, ids: Set[str]) -> Dict[str, str]:
        """Load text contexts for candidate nodes from documents.jsonl.
        
        Limits text to max_text_length to avoid token explosion.
        """
        contexts = {}
        if not os.path.exists(self.docs_path): return {}
        
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if doc['id'] in ids:
                        contexts[doc['id']] = doc.get("text", "")[:self.max_text_length]
                except: continue
        return contexts

    # =========================================================================
    # File I/O Utilities
    # =========================================================================
    
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load all objects from JSONL file.
        
        Args:
            path: Path to JSONL file.
            
        Returns:
            List of parsed objects. Invalid lines are skipped.
        """
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding="utf-8") as f:
                for line in f:
                    try: data.append(json.loads(line))
                    except: continue
        return data

    def _load_checkpoint_ids(self) -> Set[str]:
        """Load the set of already processed node IDs from checkpoint."""
        ids = set()
        if os.path.exists(self.checkpoint_path):
            logger.info(f"🔄 Checkpoint found: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r', encoding="utf-8") as f:
                for line in f:
                    try: ids.add(json.loads(line)['id'])
                    except: continue
        return ids

    def _consolidate_results(self, original_nodes):
        """Merge validated nodes with original nodes and save final output."""
        corrections = {}
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding="utf-8") as f:
                for line in f:
                    try: 
                        n = json.loads(line)
                        corrections[n['id']] = n
                    except: continue
        
        final_nodes = []
        for node in original_nodes:
            final_nodes.append(corrections.get(node['id'], node))
        
        with open(self.final_path, 'w', encoding="utf-8") as f:
            for n in final_nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        logger.info(f"💾 Validated nodes saved to {self.final_path}")