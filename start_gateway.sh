#!/bin/bash
cd /home/elwalid/projects/parallax_project
source DR-Tulu/agent/.venv/bin/activate
export PYTHONPATH="/home/elwalid/projects/parallax_project/DR-Tulu/agent:$PYTHONPATH"
set -a
source DR-Tulu/agent/.env
set +a
export GEMINI_API_KEY="$GOOGLE_AI_API_KEY"
exec python dr_tulu_agent_server.py
