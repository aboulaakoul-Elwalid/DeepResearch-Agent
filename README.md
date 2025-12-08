# Parallax Deep Research Agent

<p align="center">
<img src="parallax/src/frontend/src/assets/gradient-icon.svg" alt="Parallax" width="60" height="60">
</p>

<p align="center">
<strong>Deep research agent powered by Parallax distributed inference</strong>
</p>

<p align="center">
<img src="docs/media/architecture.png" alt="Architecture" width="800">
</p>

---

## Quick Start

### Prerequisites

- Python 3.11+
- GPU with [Parallax](https://github.com/gradient-ai/parallax) running on `localhost:3001`
- Docker (optional, for WebUI)

### 1. Clone and Setup

```bash
git clone https://github.com/aboulaakoul-elwalid/DeepResearch-Agent.git
cd DeepResearch-Agent
make setup
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
SERPER_API_KEY=your_key    # https://serper.dev
JINA_API_KEY=your_key      # https://jina.ai
EXA_API_KEY=your_key       # https://exa.ai
S2_API_KEY=your_key        # https://semanticscholar.org
```

### 3. Start the Agent

```bash
make run
```

This will show:

```
  ╔════════════════════════════════════════════════╗
  ║       Parallax Deep Research Agent             ║
  ╠════════════════════════════════════════════════╣
  ║                                                ║
  ║  WebUI:  http://localhost:3005                 ║
  ║  CLI:    make cli                              ║
  ║                                                ║
  ╚════════════════════════════════════════════════╝
```

### 4. Use the Agent

**Option A: WebUI**
1. Run `make run-webui` to start Open WebUI
2. Open [http://localhost:3005](http://localhost:3005)
3. Select `dr-tulu` or `dr-tulu-quick` model

**Option B: CLI**
```bash
make cli
```

---

## Features

- **Deep Research**: 5-15 tool calls for comprehensive answers
- **Quick Mode**: 1-5 tool calls for fast responses
- **Visible Tool Execution**: See search, browse, and synthesis in real-time
- **Citations**: Clickable source links in bibliography
- **RAG Support**: Add your own document collections as searchable knowledge bases

---

## Add Your Own Documents (RAG)

Turn any documents into a searchable knowledge base that DR-Tulu can query.

### Quick Path (3 steps)

```bash
cd rag/

# 1. Create chunks from your documents
python -c "
import json
from pathlib import Path
for f in Path('../my_docs').glob('*.txt'):
    text = f.read_text()
    for i, chunk in enumerate(text.split('\n\n')):
        if len(chunk) > 100:
            print(json.dumps({'chunk_id': f'{f.stem}_{i}', 'book_id': f.stem, 'book_title': f.stem, 'text': chunk}, ensure_ascii=False))
" > output/chunks/my_docs.jsonl

# 2. Generate embeddings
python embed_chunks.py \
    --chunks-file output/chunks/my_docs.jsonl \
    --parquet-file output/embeddings/my_docs.parquet \
    --normalize

# 3. Ingest into ChromaDB
python ingest_chroma.py \
    --inputs output/embeddings \
    --chroma-path ./chroma_db \
    --collection my_knowledge_base
```

Then set in `.env`:
```bash
ARABIC_BOOKS_CHROMA_PATH=/path/to/chroma_db
ARABIC_BOOKS_COLLECTION=my_knowledge_base
```

**Full documentation:** See [rag/README.md](rag/README.md) for:
- PDF and Markdown extraction
- Creating custom MCP tools
- Embedding model recommendations
- Troubleshooting

---

## Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies and create `.env` |
| `make run` | Start the gateway |
| `make cli` | Interactive CLI for deep research |
| `make run-webui` | Start Open WebUI container |
| `make status` | Check service status |
| `make stop` | Stop all services |

---

## Project Structure

```
DeepResearch-Agent/
├── dr_tulu_cli_gateway.py    # Gateway server
├── Makefile                  # Build commands
├── .env.example              # Environment template
│
├── DR-Tulu/agent/            # Agent code
│   ├── scripts/              # CLI scripts
│   └── workflows/            # Configuration files
│
├── parallax/                 # Parallax inference engine
│
└── rag/                      # RAG pipeline (optional)
```

---

## Troubleshooting

**Gateway not responding?**
```bash
make status
```

**Parallax not running?**
Make sure Parallax is running on `localhost:3001` before starting the agent.

**Tool calls failing?**
Check your API keys in `.env`.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
