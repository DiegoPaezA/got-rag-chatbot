"""CLI entrypoint to test hybrid retrieval and generation."""
import logging
from src.utils.logger import setup_logging
from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator

setup_logging()

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

    print("\n💬 Ask the Maester (type 'exit' to quit):")

    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break

        print("🔍 Searching archives...")
        context = retriever.retrieve(question)

        print("✍️  Writing response...")
        answer = generator.generate_answer(question, context)

        print(f"\n📜 Maester: {answer}")


if __name__ == "__main__":
    main()