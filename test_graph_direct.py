from src.rag.graph_search import GraphSearcher
import json

def main():
    print("🕸️  Iniciando Pruebas Unitarias del Grafo...\n")
    
    # Instanciamos solo el buscador de grafos
    searcher = GraphSearcher()
    
    # Batería de preguntas estructurales
    test_questions = [
        "Who is the father of Arya Stark?",               # Relación Simple
        "Which house does Tyrion Lannister belong to?",   # Pertenencia
        "What weapon does Ned Stark own?",                # Propiedad
        "Who killed Aerys II Targaryen?",                 # Acción (Eventos)
        "Who are the children of Catelyn Stark?"          # Relación inversa (Madre -> Hijos)
    ]
    
    for i, q in enumerate(test_questions, 1):
        print(f"🔹 PREGUNTA {i}: {q}")
        
        # 1. Generación de Cypher
        cypher = searcher.generate_cypher(q)
        print(f"   📝 Cypher: {cypher}")
        
        # 2. Ejecución
        results = searcher.run_query(q)
        
        if results:
            # Imprimimos bonito el JSON
            print(f"   ✅ Resultado: {json.dumps(results, indent=2, ensure_ascii=False)}")
        else:
            print("   ⚠️  Resultado VACÍO (Revisar datos o query)")
        
        print("-" * 50)

if __name__ == "__main__":
    main()