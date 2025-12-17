import sys
import os
# Hack to import modules from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.graph_search import GraphSearcher

def audit():
    print("🔍 NEO4J AUDIT")
    print("=====================")
    searcher = GraphSearcher()
    driver = searcher._get_driver()
    
    with driver.session() as session:
        # 1. Count nodes
        print("\n📊 Node counts:")
        result = session.run("MATCH (n) RETURN labels(n) as label, count(*) as count")
        for record in result:
            print(f"   - {record['label'][0]}: {record['count']}")

        # 2. List relationship types
        print("\n🔗 Existing relationship types:")
        result = session.run("CALL db.relationshipTypes()")
        rels = [r[0] for r in result]
        if not rels:
            print("   ⚠️  NO RELATIONSHIPS FOUND (the graph contains isolated nodes only)")
        else:
            for r in rels:
                print(f"   - {r}")
                
        # 3. Sample relationships
        print("\n👀 Relationship examples (Top 10):")
        result = session.run("MATCH (a)-[r]->(b) RETURN a.id, type(r), b.id LIMIT 10")
        for record in result:
            print(f"   {record['a.id']} --[{record['type(r)']}]--> {record['b.id']}")

    searcher.close()

if __name__ == "__main__":
    audit()