import os
import json
import logging
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv
from src.config_manager import ConfigManager

logger = logging.getLogger("Neo4jLoader")

class Neo4jLoader:
    """Load graph nodes and edges into Neo4j from JSONL sources."""

    def __init__(self, data_dir: str):
        """Initialize the loader with database connection and source paths.

        Environment variables used (with defaults):
        - NEO4J_URI (default: bolt://localhost:7687)
        - NEO4J_USER (default: neo4j)
        - NEO4J_PASSWORD (default: password)

        Args:
            data_dir: Directory containing node and edge JSONL files.
        """
        load_dotenv()

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Get batch sizes from config
        config_manager = ConfigManager()
        db_config = config_manager.get("database", "neo4j", default={})
        batch_config = db_config.get("batch_sizes", {})
        self.batch_size_nodes = batch_config.get("nodes", 1000)
        self.batch_size_edges = batch_config.get("edges", 1000)
        self.timeout = db_config.get("timeout", 30)

        # Prefer validated nodes; fall back to raw nodes if absent
        self.nodes_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        if not os.path.exists(self.nodes_path):
            self.nodes_path = os.path.join(data_dir, "nodes_validated.jsonl")

        self.edges_path = os.path.join(data_dir, "edges.jsonl")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def clean_db(self):
        """Delete all nodes and relationships to start from a clean slate."""
        logger.warning("🗑️  Wiping Neo4j Database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        """Create uniqueness constraints per node label based on the `id` property."""
        logger.info("⚡ Creating Indexes & Constraints...")

        # Discover node labels present in the source data
        node_types = set()
        if os.path.exists(self.nodes_path):
            with open(self.nodes_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        node_types.add(json.loads(line)['type'])
                    except Exception:
                        continue
        
        with self.driver.session() as session:
            for label in node_types:
                clean_label = "".join(x for x in label if x.isalnum())
                if not clean_label:
                    continue

                try:
                    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{clean_label}) REQUIRE n.id IS UNIQUE"
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Could not create constraint for {clean_label}: {e}")

    def load_nodes(self):
        """Bulk load nodes grouped by type."""
        logger.info("📦 Loading Nodes...")
        
        if not os.path.exists(self.nodes_path):
            logger.error(f"❌ Nodes file not found: {self.nodes_path}")
            return

        nodes_by_type = {}
        with open(self.nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ntype = "".join(x for x in data['type'] if x.isalnum())
                    
                    if ntype not in nodes_by_type:
                        nodes_by_type[ntype] = []
                    
                    # 1. Load base properties
                    props = data.get('properties', {}).copy()

                    # 2. Overwrite with normalized relations if present
                    if 'normalized_relations' in data:
                        props.update(data['normalized_relations'])

                    # 3. Convert all values to strings (fully flattened)
                    for key, val in list(props.items()):
                        if isinstance(val, list):
                            # Empty list -> empty string
                            if not val:
                                props[key] = ""
                            else:
                                # Join list items with commas
                                # Example: ["Rhaena", "Jaehaerys"] -> "Rhaena, Jaehaerys"
                                props[key] = ", ".join(str(x) for x in val)
                    
                    # 4. Ensure required fields
                    props['id'] = data['id']
                    props['name'] = props.get('name', data['id']) 
                    props['url'] = data.get('url', '')
                    props['confidence'] = data.get('confidence', 'Unknown')
                    
                    nodes_by_type[ntype].append(props)
                except Exception:
                    continue

        with self.driver.session() as session:
            for ntype, nodes in nodes_by_type.items():
                logger.info(f"   ➡️  Type '{ntype}': {len(nodes)} nodes")
                
                for i in tqdm(range(0, len(nodes), self.batch_size_nodes), desc=f"Pushing {ntype}"):
                    batch = nodes[i : i + self.batch_size_nodes]
                    
                    query = f"""
                    UNWIND $batch AS row
                    MERGE (n:{ntype} {{id: row.id}})
                    SET n += row
                    """
                    session.run(query, batch=batch)

    def load_edges(self):
        """Bulk load relationships grouped by relation type."""
        logger.info("🔗 Loading Edges...")
        
        if not os.path.exists(self.edges_path):
            logger.error(f"❌ Edges file not found: {self.edges_path}")
            return
        
        edges_by_rel = {}
        count = 0
        with open(self.edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    rel = "".join(x for x in data['relation'] if x.isalnum() or x == '_')
                    
                    if rel not in edges_by_rel:
                        edges_by_rel[rel] = []
                    
                    edges_by_rel[rel].append({
                        "source": data['source'],
                        "target": data['target']
                    })
                    count += 1
                except Exception:
                    continue
        
        logger.info(f"   ➡️  Total Edges to load: {count}")

        with self.driver.session() as session:
            for rel_type, edges in edges_by_rel.items():
                
                for i in tqdm(range(0, len(edges), self.batch_size_edges), desc=f"Linking {rel_type}"):
                    batch = edges[i : i + self.batch_size_edges]
                    
                    query = f"""
                    UNWIND $batch AS row
                    MATCH (source {{id: row.source}})
                    MATCH (target {{id: row.target}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    """
                    session.run(query, batch=batch)

    def run(self):
        """Execute the full load pipeline: clean DB, add constraints, load nodes, then edges."""
        try:
            self.clean_db()
            self.create_constraints()
            self.load_nodes()
            self.load_edges()
            logger.info("✅ Neo4j Ingestion Complete!")
        except Exception as e:
            logger.error(f"❌ Critical Error in Neo4j Loader: {e}")
        finally:
            self.close()

if __name__ == "__main__":
    # Direct run
    loader = Neo4jLoader(data_dir="data/processed")
    loader.run()