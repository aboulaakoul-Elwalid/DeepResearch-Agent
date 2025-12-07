# Parallax Deep Research Agent - Makefile
# ========================================

SHELL := /bin/bash

.PHONY: setup install run cli start-webui run-gateway run-webui run-all stop status clean help

# Configuration
PYTHON := python3
VENV := DR-Tulu/agent/.venv
GATEWAY_PORT := 3002
WEBUI_PORT := 3005

# Default target
help:
	@echo "Parallax Deep Research Agent"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Full setup (venv, deps, config)"
	@echo "  make install      - Install Python dependencies only"
	@echo ""
	@echo "Running:"
	@echo "  make run-gateway  - Start the CLI gateway (port $(GATEWAY_PORT))"
	@echo "  make run-webui    - Start Open WebUI container (port $(WEBUI_PORT))"
	@echo "  make run-all      - Start gateway + Open WebUI"
	@echo ""
	@echo "Management:"
	@echo "  make stop         - Stop all services"
	@echo "  make status       - Show running services"
	@echo "  make logs         - Show gateway logs"
	@echo "  make clean        - Remove containers and volumes"
	@echo ""
	@echo "Testing:"
	@echo "  make test-gateway - Test gateway health"
	@echo "  make test-query   - Run a test query"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

setup: check-deps create-venv install create-env
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env and configure your Parallax endpoint"
	@echo "  2. Run 'make run-all' to start services"
	@echo "  3. Open http://localhost:$(WEBUI_PORT) in your browser"
	@echo ""

check-deps:
	@echo "Checking dependencies..."
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Python 3 is required but not installed."; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }
	@echo "  Python: OK"
	@echo "  Docker: OK"

create-venv:
	@echo "Creating Python virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "  Virtual environment created at $(VENV)"; \
	else \
		echo "  Virtual environment already exists"; \
	fi

install:
	@echo "Installing Python dependencies..."
	@$(VENV)/bin/pip install --upgrade pip -q
	@$(VENV)/bin/pip install -e DR-Tulu/agent -q
	@$(VENV)/bin/pip install fastapi uvicorn chromadb sentence-transformers -q
	@echo "  Dependencies installed"

create-env:
	@echo "Creating .env file..."
	@if [ ! -f ".env" ]; then \
		cp .env.example .env 2>/dev/null || \
		echo '# Parallax Deep Research Agent Configuration\n\n# Parallax Inference Endpoint\n# Local (with GPU):\nPARALLAX_BASE_URL=http://localhost:3001/v1\n# Hosted (Modal - no GPU needed):\n# PARALLAX_BASE_URL=https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1\n\n# API Keys\nGOOGLE_AI_API_KEY=\nSERPER_API_KEY=\n\n# RAG Configuration\nARABIC_BOOKS_CHROMA_PATH=/home/elwalid/projects/parallax_project/chroma_db\nARABIC_BOOKS_COLLECTION=arabic_books' > .env; \
		echo "  .env file created - please edit it with your settings"; \
	else \
		echo "  .env file already exists"; \
	fi

# ============================================================================
# Running Services
# ============================================================================

# Main entry point - starts WebUI (if docker available) and gateway
run: start-webui
	@echo ""
	@echo "  ╔════════════════════════════════════════════════╗"
	@echo "  ║       Parallax Deep Research Agent             ║"
	@echo "  ╠════════════════════════════════════════════════╣"
	@echo "  ║                                                ║"
	@echo "  ║  WebUI:  http://localhost:$(WEBUI_PORT)               ║"
	@echo "  ║  CLI:    make cli                              ║"
	@echo "  ║                                                ║"
	@echo "  ╚════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Starting gateway... (Ctrl+C to stop)"
	@echo ""
	@cd $(CURDIR) && \
		source $(VENV)/bin/activate && \
		export PYTHONPATH="$(CURDIR)/DR-Tulu/agent:$$PYTHONPATH" && \
		set -a && source DR-Tulu/agent/.env 2>/dev/null; set +a && \
		$(PYTHON) dr_tulu_cli_gateway.py

# Auto-start WebUI container if docker is available
start-webui:
	@if command -v docker >/dev/null 2>&1; then \
		if docker ps --filter "name=open-webui" --format "{{.Names}}" | grep -q open-webui; then \
			echo "  WebUI already running"; \
		else \
			echo "  Starting WebUI container..."; \
			docker start open-webui 2>/dev/null || \
			docker run -d \
				--name open-webui \
				-p $(WEBUI_PORT):8080 \
				--add-host=host.docker.internal:host-gateway \
				-v open-webui-data:/app/backend/data \
				-e WEBUI_NAME="Parallax Deep Research" \
				-e OPENAI_API_BASE_URLS=http://host.docker.internal:$(GATEWAY_PORT)/v1 \
				-e OPENAI_API_KEYS=dummy-key \
				-e ENABLE_SIGNUP=true \
				--restart unless-stopped \
				ghcr.io/open-webui/open-webui:main >/dev/null 2>&1; \
			echo "  WebUI started"; \
		fi \
	fi

# Interactive CLI for deep research
cli:
	@cd $(CURDIR) && \
		source $(VENV)/bin/activate && \
		export PYTHONPATH="$(CURDIR)/DR-Tulu/agent:$$PYTHONPATH" && \
		set -a && source DR-Tulu/agent/.env 2>/dev/null; set +a && \
		$(PYTHON) DR-Tulu/agent/scripts/interactive_auto_search.py \
			--config DR-Tulu/agent/workflows/auto_search_modal_deep.yaml

run-gateway:
	@echo "Starting CLI Gateway on port $(GATEWAY_PORT)..."
	@cd $(CURDIR) && \
		source $(VENV)/bin/activate && \
		export PYTHONPATH="$(CURDIR)/DR-Tulu/agent:$$PYTHONPATH" && \
		set -a && source DR-Tulu/agent/.env 2>/dev/null; set +a && \
		$(PYTHON) dr_tulu_cli_gateway.py

run-webui:
	@echo "Starting Open WebUI on port $(WEBUI_PORT)..."
	@docker run -d \
		--name open-webui \
		-p $(WEBUI_PORT):8080 \
		--add-host=host.docker.internal:host-gateway \
		-v open-webui-data:/app/backend/data \
		-e WEBUI_NAME="Parallax Deep Research" \
		-e OPENAI_API_BASE_URLS=http://host.docker.internal:$(GATEWAY_PORT)/v1 \
		-e OPENAI_API_KEYS=dummy-key \
		-e ENABLE_SIGNUP=true \
		--restart unless-stopped \
		ghcr.io/open-webui/open-webui:main 2>/dev/null || \
		docker start open-webui
	@echo "  Open WebUI available at http://localhost:$(WEBUI_PORT)"

run-all:
	@echo "Starting all services..."
	@make run-webui
	@echo ""
	@echo "Starting gateway in foreground (Ctrl+C to stop)..."
	@echo "Open WebUI: http://localhost:$(WEBUI_PORT)"
	@echo ""
	@make run-gateway

# ============================================================================
# Management
# ============================================================================

stop:
	@echo "Stopping services..."
	@docker stop open-webui 2>/dev/null || true
	@pkill -f "dr_tulu_cli_gateway.py" 2>/dev/null || true
	@echo "  Services stopped"

status:
	@echo "Service Status"
	@echo "=============="
	@echo ""
	@echo "Gateway (port $(GATEWAY_PORT)):"
	@curl -s http://localhost:$(GATEWAY_PORT)/v1/models >/dev/null 2>&1 && \
		echo "  Status: RUNNING" || echo "  Status: STOPPED"
	@echo ""
	@echo "Open WebUI (port $(WEBUI_PORT)):"
	@docker ps --filter "name=open-webui" --format "  Status: {{.Status}}" 2>/dev/null || echo "  Status: STOPPED"
	@echo ""

logs:
	@echo "Gateway logs (last 50 lines):"
	@tail -50 gateway.log 2>/dev/null || echo "No log file found. Gateway may not have been started yet."

clean:
	@echo "Cleaning up..."
	@docker stop open-webui 2>/dev/null || true
	@docker rm open-webui 2>/dev/null || true
	@docker volume rm open-webui-data 2>/dev/null || true
	@echo "  Containers and volumes removed"

# ============================================================================
# Testing
# ============================================================================

test-gateway:
	@echo "Testing gateway..."
	@curl -s http://localhost:$(GATEWAY_PORT)/v1/models | $(PYTHON) -m json.tool && \
		echo "Gateway is healthy!" || echo "Gateway is not responding"

test-query:
	@echo "Running test query..."
	@curl -s -X POST http://localhost:$(GATEWAY_PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "dr-tulu-quick", "messages": [{"role": "user", "content": "What is 2+2?"}], "stream": false}' | \
		$(PYTHON) -m json.tool
