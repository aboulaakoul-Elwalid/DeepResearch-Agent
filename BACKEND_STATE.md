# Backend State (DR-Tulu + UI wiring)

Last updated: now

## What’s running
- Primary model: **Gemini** via `gemini/gemini-2.0-flash` (Modal/Parallax endpoint currently down).
- DR-Tulu configs:
  - `DR-Tulu/agent/workflows/auto_search_deep.yaml` — deep research (tool_calls=20, browse on, long_form prompt).
  - `DR-Tulu/agent/workflows/auto_search_basic.yaml` — fast/light search.
  - `DR-Tulu/agent/workflows/auto_search_parallax.yaml` — uses Modal if up, otherwise falls back to Gemini.
- Gateway for the UI: `DR-Tulu/agent/scripts/openai_gateway.py` (OpenAI-compatible, adds `/cluster/status` for Vite health).

## How to start the backend for the UI
```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent
source activate.sh  # loads GOOGLE_AI_API_KEY
GOOGLE_API_KEY=$GOOGLE_AI_API_KEY LITELLM_MODEL=gemini-2.0-flash uvicorn scripts.openai_gateway:app --host 0.0.0.0 --port 3001
```
- This exposes:
  - `GET /cluster/status` → 200
  - `GET /v1/models` → lists the Gemini model
  - `POST /v1/chat/completions` (streaming supported)

If Modal/Parallax comes back, point the UI endpoint to `https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run` (OpenAI-compatible) and stop the local gateway.

## UI targeting
- Dev UI: `http://localhost:5173/chat.html#/chat`
- Set “Endpoint” in the UI to `http://localhost:3001` when running the gateway above.
- The Vite proxy health check (`/cluster/status`) will pass with this gateway.

## Known issues / assumptions
- Network: external calls (Gemini, Serper, S2, Exa) require outbound DNS/HTTPS.
- Arabic Chroma tool: needs `chromadb` installed and the collection path set; currently disabled in configs.
- If you need DR-Tulu traces (tool_call messages) instead of raw Gemini, we should extend the gateway to wrap the `auto_search_deep` workflow; current gateway is a thin litellm front for the UI to unblock. 
