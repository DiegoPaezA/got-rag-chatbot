import sys
import os
# Hack para importar módulos desde la raíz
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.graph_search import GraphSearcher

def audit():
    print("🔍 AUDITORÍA DE NEO4J")
    print("=====================")
    searcher = GraphSearcher()
    driver = searcher._get_driver()
    
    with driver.session() as session:
        # 1. Contar Nodos
        print("\n📊 Conteo de Nodos:")
        result = session.run("MATCH (n) RETURN labels(n) as label, count(*) as count")
        for record in result:
            print(f"   - {record['label'][0]}: {record['count']}")

        # 2. Listar Tipos de Relaciones (LA CLAVE)
        print("\n🔗 Tipos de Relaciones Existentes:")
        result = session.run("CALL db.relationshipTypes()")
        rels = [r[0] for r in result]
        if not rels:
            print("   ⚠️  ¡NO HAY RELACIONES! (Tu grafo son solo puntos aislados)")
        else:
            for r in rels:
                print(f"   - {r}")
                
        # 3. Muestra de relaciones reales
        print("\n👀 Ejemplos de Relaciones (Top 10):")
        result = session.run("MATCH (a)-[r]->(b) RETURN a.id, type(r), b.id LIMIT 10")
        for record in result:
            print(f"   {record['a.id']} --[{record['type(r)']}]--> {record['b.id']}")

    searcher.close()

if __name__ == "__main__":
    audit()