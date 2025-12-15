import argparse
import json
import os
import sys
import logging
from typing import List, Dict

# Add repository root to sys.path so we can import sibling modules under `src`
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.logger import setup_logging
from src.graph.validator import GraphValidator
from src.graph.edge_builder import EdgeBuilder

setup_logging()
logger = logging.getLogger("TypeFixer")

class TypeFixer(GraphValidator):
    """Specialized validator that targets a single node type.

    This class reuses `GraphValidator` but narrows validation to nodes whose
    current `type` equals `target_type`. It writes to a temporary checkpoint so
    the main validator history remains untouched, allowing focused re-validation
    without side effects.
    """
    def __init__(self, data_dir: str, target_type: str, config_path: str = "cfg/config.json"):
        super().__init__(data_dir, config_path)
        self.target_type = target_type
        # Use a temporary checkpoint to avoid polluting the main one
        self.checkpoint_path = os.path.join(data_dir, f"fix_{target_type}_checkpoint.jsonl")

    def _should_validate(self, node: Dict) -> bool:
        """Return True if the node should be validated in this targeted pass.

        Currently validates nodes whose `type` matches `self.target_type`.
        Adjust this logic if you want to include additional heuristics
        (e.g., suspected misclassifications).
        """
        return node.get("type") == self.target_type

    def _load_checkpoint_ids(self):
        """Load checkpoint IDs from the temporary checkpoint file.

        Overrides the parent to ensure this tool uses its own checkpoint,
        effectively ignoring the primary validator history to force re-validation.
        """
        return super()._load_checkpoint_ids()

def main():
    """CLI entry point to re-validate a specific node type and optionally rebuild edges."""
    parser = argparse.ArgumentParser(description="Re-validate a specific node type and update graph.")
    parser.add_argument("--type", type=str, required=True, help="The node type to target (e.g., Creature)")
    parser.add_argument("--update-edges", action="store_true", help="Regenerate edges after validation")
    
    args = parser.parse_args()
    
    DATA_DIR = "data/processed"
    
    logger.info(f"🔧 Starting targeted re-validation for type: '{args.type}'")

    # 1) Run targeted validation
    fixer = TypeFixer(data_dir=DATA_DIR, target_type=args.type)
    
    # Uses the original validator logic but filtered by `_should_validate`
    fixer.validate() 
    
    logger.info("✅ Re-validation complete.")

    # 2) Optionally regenerate edges to reflect updated node types
    if args.update_edges:
        logger.info("🔗 Regenerating edges based on updated types...")
        edge_builder = EdgeBuilder(data_dir=DATA_DIR)
        edge_builder.run()

if __name__ == "__main__":
    main()