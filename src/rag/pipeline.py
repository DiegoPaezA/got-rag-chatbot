import logging
from src.rag.retriever import HybridRetriever
from src.rag.augmenter import ContextAugmenter
from src.rag.generator import RAGGenerator

logger = logging.getLogger("MainPipeline")

class RAGPipeline:
    """High-level pipeline gluing retriever, augmenter, and generator."""

    def __init__(self, config_path="cfg/config.json"):
        # 1) Instantiate components independently
        self.retriever = HybridRetriever(config_path=config_path)
        self.augmenter = ContextAugmenter(config_path=config_path)
        self.generator = RAGGenerator(config_path=config_path)

    def run(self, user_question: str, chat_history=None):
        """Run the full RAG flow for a single user question."""
        logger.info(f"🚀 Starting RAG for: {user_question}")

        # STEP 1: Retrieval — get raw JSON and strings
        raw_context = self.retriever.retrieve(user_question, chat_history)
        refined_query = raw_context.get("refined_query", user_question)
        
        # STEP 2: Augmentation — transform raw data into an Intelligence Report
        # The augmenter uses its own LLM to narrate graph data
        narrative_context = self.augmenter.build_context(
            query=refined_query,
            vector_context=raw_context.get("vector_context", []),
            graph_context=raw_context.get("graph_context", [])
        )
        
        logger.info("📝 Intelligence Report generated.")
        # Optional: print the report to debug what the generator receives
        # print(narrative_context)

        # STEP 3: Generation — the generator only needs the report and the question
        final_answer = self.generator.generate_answer(
            question=refined_query,
            formatted_context=narrative_context
        )

        return final_answer

# --- Usage ---
if __name__ == "__main__":
    pipeline = RAGPipeline()
    answer = pipeline.run("Who succeeded Harren Hoare?")
    print("\n🤖 MAESTER'S ANSWER:\n")
    print(answer)