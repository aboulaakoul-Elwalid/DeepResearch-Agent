# Parallax Deep Research Agent

<p align="center">
  <img src="parallax/src/frontend/src/assets/gradient-icon.svg" alt="Parallax" width="60" height="60">
</p>

<p align="center">
  <strong>A deep research agent powered by Parallax distributed inference</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#rag-setup">RAG Setup</a> •
  <a href="#configuration">Configuration</a>
</p>

---

## Overview

Parallax Deep Research Agent (DR-Tulu) is an agentic research assistant that performs multi-step web research with visible tool execution. It runs inside **Open WebUI** and uses **Parallax** for local or distributed LLM inference.

**Key Capabilities:**

- Multi-step deep research with 5-15 tool calls
- Visible tool execution (search, browse, synthesize)
- Citations with source links
- RAG support for private document collections
- Streaming responses with thinking/reasoning display

## Quick Start

### Prerequisites

- Python 3.11+
- Docker
- GPU (optional - can use hosted Modal endpoint)

### 1. Clone and Setup

```bash
git clone https://github.com/aboulaakoul-elwalid/DeepResearch-Agent.git
cd parallax-dr-tulu
# Run setup
make setup
```

### 2. Configure Parallax Endpoint

Edit `.env` and set your Parallax endpoint:

```bash
# Option A: Local Parallax (if you have a GPU)
PARALLAX_BASE_URL=http://localhost:3001/v1

# Option B: Hosted Modal endpoint
PARALLAX_BASE_URL=https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1
```

### 3. Start Services

```bash
# Start the gateway and Open WebUI
make run-all
```

### 4. Access the UI

Open [http://localhost:3005](http://localhost:3005) in your browser.

Select **dr-tulu** or **dr-tulu-quick** as your model and start researching!

---

## Features

### Research Modes

| Model           | Description         | Tool Calls | Best For                                  |
| --------------- | ------------------- | ---------- | ----------------------------------------- |
| `dr-tulu`       | Deep research mode  | 5-15       | Complex questions, comprehensive research |
| `dr-tulu-quick` | Quick research mode | 1-5        | Simple questions, fast answers            |
| `dr-tulu-deep`  | Alias for dr-tulu   | 5-15       | Same as dr-tulu                           |

### Visible Tool Execution

The agent shows its research process in real-time:

1. **Thinking** - Reasoning about the query (collapsible)
2. **Searching** - Web search with query display
3. **Browsing** - Reading and extracting from sources
4. **Synthesizing** - Combining information into a coherent answer
5. **Citations** - Clickable source links in bibliography

### RAG Support

Search private document collections using the MCP tool interface. Pre-configured with Arabic classical texts, but extensible to any document collection.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Open WebUI (localhost:3005)                  │
│              User selects "dr-tulu" model                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ POST /v1/chat/completions
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CLI Gateway (localhost:3002)                       │
│         dr_tulu_cli_gateway.py                                  │
│   - Converts CLI output to OpenAI SSE format                    │
│   - Parses <think>, <call_tool>, <answer> tags                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              DR-Tulu Agent (subprocess)                         │
│         interactive_auto_search.py                              │
│   - ReAct loop with MCP tools                                   │
│   - Web search, browse, RAG tools                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallax Inference                                 │
│   - Local: http://localhost:3001/v1                             │
│   - Hosted: Modal deployment                                    │
│   - Runs Qwen, DeepSeek, or other models                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## RAG Setup

The agent includes an MCP tool for searching private document collections stored in ChromaDB.

### How It Works

1. Documents are chunked and embedded using SentenceTransformers
2. Embeddings are stored in ChromaDB
3. The `search_arabic_books` MCP tool queries the collection
4. Results are injected into the agent's context

### Adding Your Own Documents

#### Step 1: Prepare Your Documents

Create a JSONL file with your document chunks:

```json
{"chunk_id": "doc1_chunk1", "book_id": "doc1", "book_title": "My Document", "text": "Your text content here..."}
{"chunk_id": "doc1_chunk2", "book_id": "doc1", "book_title": "My Document", "text": "More content..."}
```

#### Step 2: Generate Embeddings

```bash
python rag/embed_chunks.py \
    --chunks-file your_chunks.jsonl \
    --parquet-file output/embeddings.parquet \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

#### Step 3: Ingest into ChromaDB

```bash
python rag/ingest_chroma.py \
    --parquet-file output/embeddings.parquet \
    --chroma-path ./chroma_db \
    --collection-name your_collection
```

#### Step 4: Configure the Tool

Update `.env`:

```bash
ARABIC_BOOKS_CHROMA_PATH=/path/to/your/chroma_db
ARABIC_BOOKS_COLLECTION=your_collection
```

### Creating Custom MCP Tools

To add a new document collection as a separate tool:

1. Copy `DR-Tulu/agent/dr_agent/mcp_backend/local/arabic_books.py`
2. Rename and modify for your collection
3. Register it in the MCP tool registry
4. Add the tool to the workflow prompt

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

```bash
# Parallax Inference Endpoint
# Local (with GPU):
PARALLAX_BASE_URL=http://localhost:3001/v1
# Hosted (Modal - no GPU needed):
# PARALLAX_BASE_URL=https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1

# API Keys for web search tools
GOOGLE_AI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key

# RAG Configuration
ARABIC_BOOKS_CHROMA_PATH=/path/to/chroma_db
ARABIC_BOOKS_COLLECTION=arabic_books
```

### Workflow Configurations

Research behavior is controlled by YAML workflow configs in `DR-Tulu/agent/workflows/`:

| Config                        | Description                        |
| ----------------------------- | ---------------------------------- |
| `auto_search_modal.yaml`      | Quick mode - fewer iterations      |
| `auto_search_modal_deep.yaml` | Deep mode - comprehensive research |

### Open WebUI Settings

The gateway exposes models at `http://localhost:3002/v1`. Open WebUI connects to this endpoint.

#### Branding Customization

The Parallax "/" logo and favicon are automatically applied via volume mounts in `docker-compose.yaml`. The app name is pre-configured as "Parallax Deep Research".

**Custom assets location:** `custom-assets/`

- `favicon.png` - Browser tab icon (Parallax "/" logo)
- `logo.png` - Sidebar logo
- `qwen.png` - Qwen model icon for dr-tulu models

#### Setting Model Icons (Qwen logo for dr-tulu)

To display the Qwen logo next to dr-tulu models:

1. Start Open WebUI and log in as admin
2. Go to **Workspace > Models**
3. Click the **Edit** icon next to `dr-tulu`
4. Click on the model icon placeholder
5. Upload `custom-assets/qwen.png`
6. Click Save
7. Repeat for `dr-tulu-quick` and `dr-tulu-deep`

---

## Development

### Project Structure

```
parallax_project/
├── dr_tulu_cli_gateway.py    # Main gateway server
├── docker-compose.yaml       # Container orchestration
├── Dockerfile.gateway        # Gateway container
├── Makefile                  # Build commands
├── README.md                 # This file
│
├── custom-assets/            # Parallax branding (favicon, logo, icons)
├── scripts/                  # Setup scripts
│   └── setup.sh
│
├── DR-Tulu/                  # Agent code
│   └── agent/
│       ├── dr_agent/         # Core agent modules
│       ├── scripts/          # CLI scripts
│       └── workflows/        # Workflow configurations
│
├── parallax/                 # Parallax inference engine
│
├── rag/                      # RAG pipeline scripts
│   ├── embed_chunks.py       # Generate embeddings
│   ├── ingest_chroma.py      # Ingest into ChromaDB
│   └── ...
│
├── docs/                     # Documentation
│   └── notes/                # Development notes
│
└── deploy/                   # Deployment scripts
```

### Running Manually

```bash
# Activate the agent environment
source DR-Tulu/agent/.venv/bin/activate

# Start the gateway
python dr_tulu_cli_gateway.py

# In another terminal, start Open WebUI
docker run -d \
  --name open-webui \
  -p 3005:8080 \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:3002/v1 \
  -e OPENAI_API_KEYS=dummy-key \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

### Testing

```bash
# Test the gateway directly
curl http://localhost:3002/v1/models

# Test a completion
curl -X POST http://localhost:3002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "dr-tulu-quick", "messages": [{"role": "user", "content": "What is machine learning?"}], "stream": true}'
```

---

## Troubleshooting

### Gateway not responding

```bash
# Check if gateway is running
make status

# View gateway logs
tail -f gateway.log
```

### Open WebUI can't connect

1. Ensure gateway is running on port 3002
2. Check Docker network settings
3. Verify `host.docker.internal` resolves correctly

### Tool calls not working

1. Check API keys in `.env` (SERPER_API_KEY, GOOGLE_AI_API_KEY)
2. Verify ChromaDB path for RAG tools
3. Check agent logs for errors

---

## Credits

- **Parallax** - Distributed inference engine by Gradient
- **DR-Tulu** - Deep research agent
- **Open WebUI** - Chat interface
- **ChromaDB** - Vector database for RAG

## License

MIT License - see [LICENSE](LICENSE) for details.
