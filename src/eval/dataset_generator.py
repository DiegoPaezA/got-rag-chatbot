import json
import os
import random
import logging
import math
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURACIÓN DE ESTRATIFICACIÓN ---
# Definimos qué porcentaje de las 150 preguntas queremos por cada tipo.
# Damos peso extra a tipos "raros" pero importantes (Batallas, Objetos)
TARGET_DISTRIBUTION = {
    "Character": 0.35,      # 35% Personajes (Genealogía, Relaciones)
    "House": 0.30,          # 30% Casas (Política, Vasallaje)
    "Battle": 0.10,         # 10% Batallas (Comandantes, Lugares)
    "Location": 0.10,       # 10% Lugares (Geografía)
    "Object": 0.10,         # 10% Objetos (Armas, Dueños)
    "Organization": 0.05    # 5% Organizaciones (Guardia de la Noche, etc.)
}

# --- ESQUEMAS DE SALIDA ---
class QAItem(BaseModel):
    question: str = Field(description="A natural language question based on the node data.")
    ground_truth: str = Field(description="The precise answer derived from the node properties.")
    type: str = Field(description="The category of the question (e.g., Lineage, Politics, Geography).")
    difficulty: str = Field(description="Easy, Medium, or Hard.")

class QABatch(BaseModel):
    qa_pairs: List[QAItem]

# --- CLASE GENERADORA ---
class StratifiedDatasetGenerator:
    def __init__(self, data_dir: str, model_name: str = "gemini-2.5-flash"):
        self.nodes_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        self.output_path = os.path.join("data", "eval", "golden_dataset_150_stratified.jsonl")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.8, # Creatividad alta para fraseo variado
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=3
        )

    def _load_and_stratify_nodes(self, total_questions: int) -> List[Dict]:
        """Carga nodos y selecciona una muestra estratificada según TARGET_DISTRIBUTION."""
        
        # 1. Cargar y Agrupar
        nodes_by_type: Dict[str, List[Dict]] = {k: [] for k in TARGET_DISTRIBUTION.keys()}
        # Un bucket para "Otros" por si acaso
        nodes_by_type["Other"] = [] 

        if not os.path.exists(self.nodes_path):
            logger.error(f"❌ No se encontró {self.nodes_path}")
            return []

        logger.info("📦 Cargando y clasificando nodos...")
        with open(self.nodes_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    n = json.loads(line)
                    ntype = n.get("type", "Other")
                    
                    # Filtro de Calidad: Solo nodos con al menos algo de info útil
                    # (normalized_relations o propiedades clave)
                    has_rels = len(n.get("normalized_relations", {})) > 0
                    has_props = len(n.get("properties", {})) > 2
                    
                    if has_rels or has_props:
                        if ntype in nodes_by_type:
                            nodes_by_type[ntype].append(n)
                        else:
                            nodes_by_type["Other"].append(n)
                except:
                    continue

        # 2. Seleccionar Muestra Estratificada
        selected_nodes = []
        
        # Asumimos que generamos ~3 preguntas por nodo, así que necesitamos menos nodos
        questions_per_node_avg = 3 
        target_nodes_count = math.ceil(total_questions / questions_per_node_avg)
        
        logger.info(f"🎯 Objetivo: Seleccionar aprox {target_nodes_count} nodos para generar {total_questions} preguntas.")

        for ntype, ratio in TARGET_DISTRIBUTION.items():
            count_needed = math.ceil(target_nodes_count * ratio)
            available_nodes = nodes_by_type.get(ntype, [])
            
            if not available_nodes:
                logger.warning(f"⚠️ No hay nodos disponibles para el tipo '{ntype}'")
                continue
                
            # Si hay menos nodos de los necesarios, los tomamos todos
            count_to_take = min(count_needed, len(available_nodes))
            
            # Selección aleatoria
            sample = random.sample(available_nodes, count_to_take)
            selected_nodes.extend(sample)
            logger.info(f"   - {ntype}: Seleccionados {len(sample)} (Objetivo: {count_needed})")

        # Si faltan nodos (porque algunas categorías tenían pocos), rellenar con Characters
        if len(selected_nodes) < target_nodes_count:
            deficit = target_nodes_count - len(selected_nodes)
            chars = nodes_by_type.get("Character", [])
            # Evitar duplicados si es posible (aunque sample ya los sacó, aquí simplificamos)
            extras = random.sample(chars, min(deficit, len(chars)))
            selected_nodes.extend(extras)
            logger.info(f"   - Relleno (Characters): {len(extras)} nodos extra.")

        # Mezclar para que no queden ordenados por tipo
        random.shuffle(selected_nodes)
        return selected_nodes

    def _create_chain(self):
        """Crea la cadena de LangChain."""
        system_prompt = """
        You are an expert Game of Thrones trivia master creating an evaluation dataset.
        
        Input: JSON object of a Knowledge Graph NODE.
        Task: Generate 3 DISTINCT Question-Answer pairs based ONLY on this data.

        **Requirements:**
        1. **High Variety:** Ask about relationships (Father, Allegiance), attributes (Words, Seat, Region), or reverse facts ("Who owns Ice?").
        2. **Multi-Hop Logic (Preferred):** Instead of "Who is Robb's father?", ask "Which House rules the region where Robb Stark was born?" (if data permits).
        3. **Ground Truth:** Must be exact strings found in the JSON.
        4. **Difficulty:** Mix of Easy, Medium, Hard.

        **Input Node:**
        {node_json}
        
        Return JSON object with 'qa_pairs'.
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        parser = JsonOutputParser(pydantic_object=QABatch)
        return prompt | self.llm | parser

    def generate(self, num_questions=150):
        # 1. Obtener nodos estratificados
        nodes_to_process = self._load_and_stratify_nodes(num_questions)
        
        if not nodes_to_process:
            logger.error("❌ No se pudieron seleccionar nodos.")
            return

        chain = self._create_chain()
        dataset = []
        
        logger.info(f"🚀 Iniciando generación con LLM sobre {len(nodes_to_process)} nodos...")

        pbar = tqdm(total=num_questions)
        
        for node in nodes_to_process:
            if len(dataset) >= num_questions:
                break
                
            try:
                # Limpiamos el nodo para no gastar tokens innecesarios, pero mantenemos lo rico
                node_context = {
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "normalized_relations": node.get("normalized_relations"),
                    # Pasamos propiedades clave crudas para preguntas de atributos
                    "properties": {k:v for k,v in node.get("properties", {}).items() 
                                   if k in ["Seat", "Region", "Words", "Arms", "Founded", "Episode"]}
                }

                response = chain.invoke({"node_json": json.dumps(node_context)})
                
                if "qa_pairs" in response:
                    for item in response["qa_pairs"]:
                        if len(dataset) >= num_questions: break
                        
                        item["question_id"] = f"gen_{len(dataset)}"
                        # Añadimos metadata del nodo origen para trazabilidad
                        item["source_node_type"] = node.get("type")
                        
                        dataset.append(item)
                        pbar.update(1)
                        
            except Exception as e:
                # logger.warning(f"Error en nodo {node.get('id')}: {e}")
                continue
        
        pbar.close()

        # Guardar
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                record = item if isinstance(item, dict) else item.dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"✅ Dataset Estratificado Generado: {self.output_path}")
        
        # Reporte final de distribución
        type_counts = {}
        for item in dataset:
            t = item.get("source_node_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"📊 Distribución final de preguntas: {json.dumps(type_counts, indent=2)}")

if __name__ == "__main__":
    generator = StratifiedDatasetGenerator(data_dir="data/processed")
    generator.generate(num_questions=150)