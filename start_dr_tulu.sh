#!/bin/bash
# Start DR-Tulu Agent Server for Open WebUI
# Usage: ./start_dr_tulu.sh

cd /home/elwalid/projects/parallax_project
source DR-Tulu/agent/.venv/bin/activate

echo "Starting DR-Tulu Agent Server..."
echo "Open WebUI should connect to: http://localhost:3001/v1"
echo "Press Ctrl+C to stop"
echo ""

python dr_tulu_agent_server.py
