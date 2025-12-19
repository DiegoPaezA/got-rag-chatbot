# GOT RAG Chatbot

A sophisticated Knowledge Graph and Retrieval-Augmented Generation (RAG) system for Game of Thrones lore. This project combines web scraping, heuristic entity extraction, LLM-powered validation, and graph construction to create an intelligent, context-aware knowledge base.

## 🌟 Features

- **Intelligent Web Scraping**: Automated extraction from Game of Thrones Fandom Wiki with resume capability
- **Hybrid Knowledge Graph Construction**:
  - Heuristic-based entity extraction and type classification
  - Optional LLM validation (Gemini) for improved accuracy
  - Schema-aware relationship building with business logic constraints
- **Advanced Text Processing**: WikiText parsing, infobox extraction, and intelligent text cleaning
- **Resumable Pipeline**: Checkpoint system allows interrupted processes to resume without data loss
- **Batch Processing**: Efficient handling of large datasets with configurable batch sizes
- **Error Handling**: Robust retry logic with exponential backoff for API rate limits
- **Extensible Architecture**: Modular design with clear separation of concerns

<details>

<summary> 📁 Project Structure</summary>

```markdown
got-rag-chatbot/
│
├── .env                    # Environment variables (GOOGLE_API_KEY for Gemini)
├── .gitignore              
├── README.md
├── requirements.txt
├── pyproject.toml          # Project metadata and dependencies
│
├── cfg/                    # Configuration files
│   └── config.json         # LLM settings, prompts, and schema definitions
│
├── data/                   # Data storage (gitignored)
│   ├── raw/                
│   │   └── wiki_dump.jsonl # Raw scraped data from wiki
│   ├── processed/          # Generated knowledge graph files
│   │   ├── nodes.jsonl             # Heuristic nodes
│   │   ├── nodes_validated.jsonl   # LLM-validated nodes
│   │   ├── nodes_llm_checkpoint.jsonl # Validation checkpoint
│   │   ├── edges.jsonl             # Graph relationships
│   │   └── documents.jsonl         # Text documents for RAG
│   └── chromadb/           # Vector database (future feature)
│
├── src/                    # Main application source code
│   ├── __init__.py
│   ├── config.py           # Configuration loader
│   │
│   ├── core/               # Core components
│   │   ├── database.py     # Vector DB connection logic
│   │   └── llm.py          # LLM Client setup
│   │
│   ├── ingestion/          # Data extraction and processing
│   │   ├── scraper.py      # Fandom Wiki scraper with resume capability
│   │   ├── processor.py    # Text processing and cleaning
│   │   └── loader.py       # Database loading logic
│   │
│   ├── graph/              # Knowledge Graph construction
│   │   ├── builder.py      # Heuristic node extraction and typing
│   │   ├── validator.py    # LLM-powered node validation
│   │   └── edge_builder.py # Schema-aware relationship extraction
│   │
│   ├── utils/              # Utility functions
│   │   └── text.py         # Text cleaning and normalization
│   │
│   ├── rag/                # RAG system (future enhancement)
│   │   ├── retriever.py    
│   │   ├── chain.py        
│   │   └── engine.py       
│   │
│   ├── schemas/            # Pydantic models
│   │   ├── chat.py         
│   │   └── document.py     
│   │
│   └── api/                # FastAPI web layer (future feature)
│       ├── main.py         
│       ├── routes.py       
│       └── dependencies.py 
│
└── main.py                 # CLI orchestrator
```
</details>

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google Gemini API Key (for LLM validation feature)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/DiegoPaezA/got-rag-chatbot.git
cd got-rag-chatbot
```

2. **Install dependencies**

#### Using `uv` (Recommended)

```bash
# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

1. **Set up environment variables**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Configuration

Edit `cfg/config.json` to customize:

- LLM settings (model, temperature, retries)
- Entity types and validation prompts
- Schema constraints for relationships

## 📖 Usage

### 1. Scrape Wiki Data

Download raw data from Game of Thrones Fandom Wiki:

```bash
python main.py scrape
```

**Features:**

- Automatically resumes if interrupted
- Avoids duplicates using ID tracking
- Progress bar with real-time statistics
- Polite rate limiting (0.1s delay between requests)

**Output:** `data/raw/wiki_dump.jsonl`

### 2. Build Knowledge Graph

Extract entities and relationships from raw data:

```bash
# Build with heuristics only (fast)
python main.py build

# Build with LLM validation (more accurate, requires API key)
python main.py build --use-llm
```

**Pipeline Steps:**

#### Step 1: Heuristic Node Extraction

- Parses WikiText and extracts infoboxes
- Applies scoring system to classify entities (Character, House, Location, etc.)
- Generates confidence scores (High/Medium/Low)
- **Output:** `data/processed/nodes.jsonl`

#### Step 2: LLM Validation (Optional)

- Validates low-confidence and ambiguous nodes
- Uses Google Gemini with structured output
- Batch processing with checkpoint system
- Handles API rate limits with exponential backoff
- **Output:** `data/processed/nodes_validated.jsonl`

#### Step 3: Schema-Aware Edge Generation

- Extracts relationships from node properties
- Validates edges against schema constraints
- Prevents semantic errors (e.g., "Sword is father of Character")
- Deduplicates relationships
- **Output:** `data/processed/edges.jsonl`

## 🏗️ Architecture

### Knowledge Graph Pipeline

```markdown
┌─────────────┐
│  Wiki API   │
└──────┬──────┘
       │ Scraper (resumable)
       ▼
┌─────────────────┐
│ wiki_dump.jsonl │
└────────┬────────┘
         │ Builder (heuristic scoring)
         ▼
    ┌─────────┐         ┌──────────────┐
    │  nodes  │────────▶│  documents   │
    └────┬────┘         └──────────────┘
         │                  (for RAG)
         │ Validator (LLM)
         ▼
┌──────────────────┐
│ nodes_validated  │
└────────┬─────────┘
         │ EdgeBuilder (schema-aware)
         ▼
    ┌────────┐
    │  edges │
    └────────┘
```

### Key Components

#### 1. **FandomScraper** (`src/ingestion/scraper.py`)

- MediaWiki API integration
- Checkpoint-based resume capability
- WikiText parsing with mwparserfromhell
- Infobox extraction

#### 2. **GraphBuilder** (`src/graph/builder/builder.py`)

- Heuristic type classification using scoring system
- Property extraction and cleaning
- Document generation for RAG

#### 3. **GraphValidator** (`src/graph/validator.py`)

- LLM-powered validation with Gemini
- Batch processing with configurable size
- Checkpoint system for long-running validations
- Intelligent retry logic for API failures
- Validates nodes with:
  - Low confidence scores
  - Ambiguous types (Lore, Organization, Object)
  - Close type scores (within 2 points)

#### 4. **EdgeBuilder** (`src/graph/edge_builder.py`)

- Property-to-relationship mapping
- Schema constraint validation
- Supports multiple relationship types:
  - Family (CHILD_OF, PARENT_OF, SIBLING_OF, MARRIED_TO)
  - Loyalty (BELONGS_TO, SWORN_TO, VASSAL_OF)
  - Geography (LOCATED_IN, SEATED_AT)
  - Culture (HAS_CULTURE, FOLLOWS_RELIGION)
  - War (PARTICIPANT_IN, COMMANDED_BY)
  - Objects (OWNED_BY, WIELDED_BY, CREATED_BY)
  - Meta (PLAYED_BY, APPEARED_IN_SEASON)

### Schema Constraints

The system enforces semantic correctness through schema constraints. For example:

```python
{
    "CHILD_OF": ["Character", "Creature"],  # Only these types can have parents
    "MARRIED_TO": ["Character"],            # Only characters can marry
    "SEATED_AT": ["House"],                 # Only houses have seats
    "OWNS_WEAPON": ["Character"]            # Only characters own weapons
}
```

This prevents illogical relationships like "Ice (sword) is the father of Jon Snow".

## 📊 Data Formats

### Node Schema

```json
{
    "id": "Jon_Snow",
    "type": "Character",
    "confidence": "High (LLM)",
    "reason": "Has actor, born properties",
    "type_scores": {"Character": 5, "House": 0, ...},
    "properties": {
        "Father": "Rhaegar Targaryen",
        "House": "Stark",
        "Actor": "Kit Harington"
    },
    "url": "https://gameofthrones.fandom.com/wiki/Jon_Snow"
}
```

### Edge Schema

```json
{
    "source": "Jon_Snow",
    "relation": "CHILD_OF",
    "target": "Rhaegar_Targaryen"
}
```

### Document Schema

```json
{
    "id": "Jon_Snow",
    "text": "Jon Snow is a Character. Jon Snow is the son of...",
    "metadata": {
        "type": "Character",
        "source": "wiki_dump"
    }
}
```

## 🛠️ Advanced Features

### Resume Capability

Both scraper and validator support resuming interrupted runs:

- **Scraper**: Tracks processed article IDs in the output file
- **Validator**: Uses checkpoint file to track validated nodes

Simply re-run the same command to resume where it left off.

### Batch Processing

The validator processes nodes in configurable batches (default: 10):

- Reduces API calls
- Enables checkpoint saves between batches
- Better error recovery

### Error Handling

- **429 Rate Limits**: Exponential backoff with jitter (4s → 8s → 16s → ...)
- **Network Errors**: Automatic retry with delay
- **Parse Errors**: Graceful skip with logging
- **Schema Violations**: Tracked and reported in statistics

## 🔧 Configuration

### `cfg/config.json` Structure

```json
{
    "llm_settings": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.0,
        "max_retries": 1
    },
    "graph_settings": {
        "allowed_types": [
            "Character", "House", "Location", "Battle",
            "Object", "Creature", "Religion", "Episode",
            "Organization", "Event", "Culture", "Lore"
        ]
    },
    "prompts": {
        "validator_system": "You are an expert...",
        "validator_human": "{input_data}"
    }
}
```

## 📈 Statistics & Output

After running the build pipeline, you'll see:

```
✅ Heuristic Build Done: 2847 nodes, 2521 docs.
📊 Total: 2847. Processed: 0. To Validate: 1234
✅ Batch 1 processed (10 nodes)
...
💾 Validated nodes saved to data/processed/nodes_validated.jsonl
✅ Edges built: 4562. Skipped by Schema: 237
```

## 🚧 Future Enhancements

- [X] Vector database integration (ChromaDB)
- [X] RAG query engine
- [ ] FastAPI REST API
- [X] Neo4j export capability
- [X] Interactive chatbot interface

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Game of Thrones Fandom Wiki for the data source
- LangChain for LLM orchestration
- Google Gemini for entity validation

## 👤 Author

Diego Páez A. - [GitHub](https://github.com/DiegoPaezA)
