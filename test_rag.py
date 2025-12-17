"""CLI entrypoint to test hybrid retrieval and generation with memory."""
import logging
from src.utils.logger import setup_logging
from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator

setup_logging()

# Silenciar logs ruidosos de librerías externas
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

def main():
    """Run an interactive console chat with the Maester."""
    print("🏰 Initializing Game of Thrones RAG System...")

    retriever = HybridRetriever()
    generator = RAGGenerator()

    # --- MEMORIA DE LA SESIÓN ---
    # Almacena tuplas: ("Human", "Pregunta"), ("AI", "Respuesta")
    chat_history = [] 

    print("\n💬 Ask the Maester (type 'exit' to quit):")

    while True:
        try:
            question = input("\nYou: ")
            if question.lower() in ["exit", "quit"]:
                break

            print("🔍 Searching archives...")
            
            # 1. Pasamos el historial reciente (últimos 3 turnos)
            # El retriever se encargará de reescribir la pregunta si es necesario
            context = retriever.retrieve(question, chat_history=chat_history[-3:])

            # Mostrar si la pregunta fue reescrita (Debug útil)
            if "refined_query" in context and context["refined_query"] != question:
                print(f"   (🧠 Rewritten as: '{context['refined_query']}')")

            print("✍️  Writing response...")
            answer = generator.generate_answer(question, context)

            print(f"\n📜 Maester: {answer}")
            
            # 2. Actualizamos la memoria
            chat_history.append(("Human", question))
            chat_history.append(("AI", answer))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()