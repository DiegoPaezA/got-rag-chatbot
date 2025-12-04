# GOT RAG Chatbot

got-rag-chatbot is a Retrieval-Augmented Generation (RAG) chatbot that leverages large language models (LLMs) and vector databases to provide accurate and context-aware responses based on scraped web content.

```markdown
got-rag-chatbot/
│
├── .env                    # Environment variables (API Keys, DB path)
├── .gitignore              # Ignore venv, .env, __pycache__, etc.
├── README.md
├── requirements.txt
├── notebook_experiments/   # Jupyter notebooks for dirty testing/prototyping
│   └── 01_scraping_test.ipynb
│
├── data/                   # Local storage (ignored in git usually)
│   ├── raw/                # Raw scraped .txt or .json files
│   └── chromadb/           # The persistent Vector Database files
│
├── src/                    # Main application source code
│   ├── __init__.py
│   │
│   ├── config.py           # Configuration loader (settings, paths)
│   │
│   ├── core/               # Core components shared across the app
│   │   ├── database.py     # Vector DB connection logic (Singleton)
│   │   └── llm.py          # LLM Client setup (OpenAI/Ollama)
│   │
│   ├── ingestion/          # ETL Pipeline (Extract, Transform, Load)
│   │   ├── scraper.py      # Logic to scrape the websites
│   │   ├── processor.py    # Text splitting and cleaning logic
│   │   └── loader.py       # Logic to embed text and save to ChromaDB
│   │
│   ├── rag/                # The "Brain" (Retrieval Augmented Generation)
│   │   ├── retriever.py    # Logic to search/query the Vector DB
│   │   ├── chain.py        # LangChain logic (Prompt templates + LLM)
│   │   └── engine.py       # Main class that ties Retrieval + Generation
│   │
│   ├── schemas/            # Pydantic models (Data contracts)
│   │   ├── chat.py         # E.g., class QueryRequest(BaseModel): ...
│   │   └── document.py     # E.g., class ScrapedDocument(BaseModel): ...
│   │
│   └── api/                # Web Layer (FastAPI) - FUTURE PROOFING
│       ├── main.py         # FastAPI app instance
│       ├── routes.py       # API Endpoints
│       └── dependencies.py # Dependency injection
│
└── main.py                 # CLI Entry point (The file you run in terminal)
```