import argparse
import json
import os
import sys
import logging
from typing import List, Dict

# Truco para importar módulos hermanos (src) desde una subcarpeta
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.graph.validator import GraphValidator
from src.graph.edge_builder import EdgeBuilder

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TypeFixer")

class TypeFixer(GraphValidator):
    """
    Versión especializada del Validador que solo ataca un tipo específico
    e ignora el checkpoint histórico para forzar la re-evaluación.
    """
    def __init__(self, data_dir: str, target_type: str, config_path: str = "cfg/config.json"):
        super().__init__(data_dir, config_path)
        self.target_type = target_type
        # Usamos un checkpoint temporal para no ensuciar el principal
        self.checkpoint_path = os.path.join(data_dir, f"fix_{target_type}_checkpoint.jsonl")

    def _should_validate(self, node: Dict) -> bool:
        # Lógica Quirúrgica: Solo validar si es del tipo objetivo
        # Opcional: También podrías incluir 'Object' si sospechas que hay criaturas escondidas ahí
        return node.get("type") == self.target_type

    def _load_checkpoint_ids(self):
        # Sobreescribimos para que empiece de cero (o desde el checkpoint temporal)
        # Esto ignora el historial principal, obligando a re-validar
        return super()._load_checkpoint_ids()

def main():
    parser = argparse.ArgumentParser(description="Re-validate a specific node type and update graph.")
    parser.add_argument("--type", type=str, required=True, help="The node type to target (e.g., Creature)")
    parser.add_argument("--update-edges", action="store_true", help="Regenerate edges after validation")
    
    args = parser.parse_args()
    
    DATA_DIR = "data/processed"
    
    logger.info(f"🔧 Starting targeted repair for type: '{args.type}'")

    # 1. EJECUTAR VALIDACIÓN QUIRÚRGICA
    fixer = TypeFixer(data_dir=DATA_DIR, target_type=args.type)
    
    # Esto usará la lógica de tu Validator original pero filtrada por _should_validate
    fixer.validate() 
    
    logger.info("✅ Re-validation complete.")

    # 2. REGENERAR ARISTAS (Opcional pero recomendado)
    if args.update_edges:
        logger.info("🔗 Regenerating Edges based on new types...")
        edge_builder = EdgeBuilder(data_dir=DATA_DIR)
        edge_builder.run()

if __name__ == "__main__":
    main()