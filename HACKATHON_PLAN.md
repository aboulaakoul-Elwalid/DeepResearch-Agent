# Parallax Deep Research Agent – Hackathon Plan (Track 2 – Applications)

## Goal

Build a working **deep research agent** powered by **Parallax local inference** that:

- Runs as a **“dr-tulu” model** inside **Open WebUI**
- Shows **visible tool execution** (search/browse phases)
- Supports **RAG over local documents (Arabic books)**
- Ships as a **public OSS repo + short demo video**
- Is ready to submit to Gradient’s **“Build your own AI Lab”** campaign (Track 2 – Building Applications).

---

## Hard Deadlines (Track 2 – Building Applications)

- Campaign window: **Nov 17, 2025 – Dec 7, 2025**
- Submission deadline: **Dec 7, 2025 @ 11:59 PM US Eastern** (Google Form)
- Must share at least **one social post** (X / Reddit / etc.) showing the Parallax use case and describing how the app works.
- Strongly recommended: **GitHub repo + demo video + multiple social posts**.

(You’ll link the repo, video, and social post(s) in the submission form.)

---

## Architecture Decision (Frozen for Hackathon)

### Final Choice: **Option B – CLI-as-Backend (Subprocess Streaming)**

We **do not** fight token-limit edge cases in the web agent for this hackathon. Instead:

1. Keep the **existing CLI deep research agent** (`interactive_auto_search.py`) as the **single source of truth** for the agentic loop.
2. Write a **thin HTTP gateway** (e.g. `dr_tulu_cli_gateway.py`) that:
   - Exposes **OpenAI-compatible** `POST /v1/chat/completions` with `stream=true`
   - Spawns the CLI as a **subprocess** per request
   - Reads CLI **stdout line-by-line** and parses tags:
     - `<think>...</think>` – reasoning / phase updates
     - `<call_tool name="...">...</call_tool>` – tool calls
     - `<tool_output>...</tool_output>` – tool results
     - `<answer>...</answer>` – final answer with citations
   - Converts those into **SSE chunks** that Open WebUI understands:
     - `assistant` chunks with `delta.content` for thinking / status
     - `assistant` chunks with `delta.tool_calls[...]` for tools
     - `tool` chunks with `tool_call_id` + `content` for tool outputs
     - Final `assistant` chunk with the answer markdown.

3. **Configure the CLI** to use **Parallax local inference** (OpenAI-compatible endpoint) as its backend model.

This avoids touching the fragile `client.py` token limit logic and gives you a stable, testable core: “CLI works → gateway just mirrors it.”

---

## Success Criteria

### P0 – Must Have (Demo-Blocking)

By submission time:

- [ ] **Parallax** is running locally with at least **one LLM** suitable for research (e.g. Qwen / GLM / Llama through Parallax).
- [ ] `dr_tulu_cli_gateway.py` exposes `/v1/chat/completions` and uses **Parallax** as the underlying LLM through the CLI.
- [ ] **Open WebUI container** is healthy and accessible at `http://localhost:3005`.
- [ ] User can select `"dr-tulu"` (or `"dr-tulu-parallax"`) as a model in Open WebUI.
- [ ] Sending a query in WebUI triggers:
  - Multi-step deep research in the CLI (4–10 tool calls for your best demos)
  - **Streaming** response in WebUI (tokens appear progressively).
- [ ] Final answer in WebUI:
  - Well-structured markdown
  - **Citations** with source list (as produced by CLI).

### P1 – Should Have (Demo Enhancers)

- [ ] **Tool timeline visible** in Open WebUI:
  - Tool calls appear as `tool_calls` chunks
  - Tool results appear as `role: "tool"` messages
- [ ] **RAG**: user can query a small **Arabic books** collection via a `search_arabic_books` / RAG tool.
- [ ] Research phases visible through streaming content:
  - “Searching…”
  - “Browsing…”
  - “Synthesizing answer…”

### P2 – Nice to Have (Developer Cred / Bonus Points)

- [ ] **CLI usage documented** (how to run the agent directly from terminal).
- [ ] Simple **RAG ingestion script** to add custom documents.
- [ ] Early **MCP integration story** documented (even if not fully polished).
- [ ] Optional: minimal `docker-compose` to spin up Parallax + gateway + WebUI.

---

## Final Architecture (Hackathon Version)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Open WebUI (3005)                            │
│                  - User selects `dr-tulu` model                        │
│                  - Shows: streaming answer + tool timeline             │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            │  POST /v1/chat/completions (stream=true)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                CLI Gateway Server (3001) - `dr_tulu_cli_gateway.py`    │
│  - Accepts OpenAI-style chat completions requests                      │
│  - Spawns `interactive_auto_search.py` as subprocess                   │
│  - Parses <think>, <call_tool>, <tool_output>, <answer>                │
│  - Emits OpenAI-compatible SSE chunks with tool_calls + tool results   │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        DR-Tulu CLI Deep Research Agent (`interactive_auto_search.py`)  │
│  - Full ReAct loop, MCP tools, citations                               │
│  - Configured to use Parallax OpenAI-compatible endpoint               │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Parallax Local Inference                         │
│    - Runs your chosen LLM(s) on local GPUs                             │
│    - OpenAI-compatible endpoint (e.g. http://localhost:4000/v1)       │
└─────────────────────────────────────────────────────────────────────────┘
```
