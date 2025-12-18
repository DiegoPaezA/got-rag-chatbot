"""CLI entrypoint to test hybrid retrieval and generation with memory."""
import logging
from src.utils.logger import setup_logging
from src.rag.retriever import HybridRetriever
from src.rag.augmenter import ContextAugmenter
from src.rag.generator import RAGGenerator

setup_logging()

# Silence noisy logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

def main():
    """Run an interactive console chat with the Maester."""
    print("🏰 Initializing Game of Thrones RAG System...")

    # Initialize decoupled components
    retriever = HybridRetriever()
    augmenter = ContextAugmenter()  # Loads its own internal LLM
    generator = RAGGenerator()      # Loads its own internal LLM

    # --- SESSION MEMORY ---
    # Stores tuples: ("Human", question), ("AI", answer)
    chat_history = [] 

    print("\n💬 Ask the Maester (type 'exit' to quit):")

    while True:
        try:
            question = input("\nYou: ")
            if question.lower() in ["exit", "quit"]:
                break

            print("🔍 Searching archives...")
            
            # 1. RETRIEVAL
            # Pass recent history for context (coreference resolution)
            raw_context = retriever.retrieve(question, chat_history=chat_history[-3:])
            
            # IMPORTANT: Use refined query (e.g., "Who is he?" -> "Who is Tyrion?")
            refined_query = raw_context.get("refined_query", question)

            # Show if the question was rewritten (useful debug)
            if refined_query != question:
                print(f"   (🧠 Rewritten as: '{refined_query}')")

            # 2. AUGMENTATION (Analyst)
            # Transform raw data into an Intelligence Report.
            # Critical: pass 'query=refined_query' so the LLM interprets correctly.
            print("📝 Compiling intelligence report...")
            formatted_context_str = augmenter.build_context(
                query=refined_query, 
                vector_context=raw_context.get("vector_context", []),
                graph_context=raw_context.get("graph_context", [])
            )

            # 3. GENERATION (Maester)
            # Generate the final answer based on the report.
            print("✍️  Writing response...")
            answer = generator.generate_answer(
                question=refined_query, 
                formatted_context=formatted_context_str
            )

            print(f"\n📜 Maester: {answer}")
            
            # 4. Update memory
            # Keep the user's original question to preserve the natural flow
            chat_history.append(("Human", question))
            chat_history.append(("AI", answer))

        except KeyboardInterrupt:
            print("\nValar Morghulis.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()