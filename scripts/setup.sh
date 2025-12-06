#!/bin/bash
# Parallax Deep Research Agent - Setup Script
# ============================================
#
# This script sets up the Parallax Deep Research Agent environment.
# It creates a Python virtual environment, installs dependencies,
# and configures the necessary settings.
#
# Usage:
#   ./scripts/setup.sh
#
# Options:
#   --modal    Configure for Modal hosted endpoint (no local GPU needed)
#   --local    Configure for local Parallax endpoint (requires GPU)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/DR-Tulu/agent/.venv"
ENV_FILE="$PROJECT_ROOT/.env"
AGENT_ENV_FILE="$PROJECT_ROOT/DR-Tulu/agent/.env"

# Default endpoint
DEFAULT_LOCAL_ENDPOINT="http://localhost:3001/v1"
MODAL_ENDPOINT="https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       Parallax Deep Research Agent - Setup                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
USE_MODAL=false
USE_LOCAL=false
for arg in "$@"; do
    case $arg in
        --modal)
            USE_MODAL=true
            ;;
        --local)
            USE_LOCAL=true
            ;;
    esac
done

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Step 1: Check dependencies
echo ""
echo "Step 1: Checking dependencies..."
echo "─────────────────────────────────"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not installed"
    echo "  Install Python 3.11+ and try again"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version 2>&1 | cut -d' ' -f3 | tr -d ',')
    print_status "Docker $DOCKER_VERSION found"
else
    print_error "Docker is required but not installed"
    echo "  Install Docker and try again"
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    print_status "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    echo "  Start Docker and try again"
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Setting up Python environment..."
echo "─────────────────────────────────────────"

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_PATH" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    print_status "Virtual environment created at $VENV_PATH"
else
    print_status "Virtual environment already exists"
fi

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
echo "───────────────────────────────────"

print_info "Upgrading pip..."
"$VENV_PATH/bin/pip" install --upgrade pip -q

print_info "Installing agent dependencies..."
"$VENV_PATH/bin/pip" install -e DR-Tulu/agent -q 2>/dev/null || \
    "$VENV_PATH/bin/pip" install -r DR-Tulu/agent/requirements.txt -q 2>/dev/null || true

print_info "Installing gateway dependencies..."
"$VENV_PATH/bin/pip" install fastapi uvicorn chromadb sentence-transformers -q

print_status "Dependencies installed"

# Step 4: Configure environment
echo ""
echo "Step 4: Configuring environment..."
echo "────────────────────────────────────"

# Determine endpoint
if [ "$USE_MODAL" = true ]; then
    ENDPOINT="$MODAL_ENDPOINT"
    print_info "Using Modal hosted endpoint (no GPU required)"
elif [ "$USE_LOCAL" = true ]; then
    ENDPOINT="$DEFAULT_LOCAL_ENDPOINT"
    print_info "Using local Parallax endpoint (GPU required)"
else
    # Ask user
    echo ""
    echo "Select your Parallax endpoint:"
    echo ""
    echo "  1) Local (requires GPU running Parallax on port 3001)"
    echo "  2) Modal hosted (no GPU needed, uses cloud inference)"
    echo ""
    read -p "Enter choice [1/2] (default: 2): " choice
    
    case $choice in
        1)
            ENDPOINT="$DEFAULT_LOCAL_ENDPOINT"
            print_info "Using local Parallax endpoint"
            ;;
        *)
            ENDPOINT="$MODAL_ENDPOINT"
            print_info "Using Modal hosted endpoint"
            ;;
    esac
fi

# Create .env file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << EOF
# Parallax Deep Research Agent Configuration
# ==========================================

# Parallax Inference Endpoint
PARALLAX_BASE_URL=$ENDPOINT

# API Keys (required for web search tools)
GOOGLE_AI_API_KEY=
SERPER_API_KEY=

# RAG Configuration
ARABIC_BOOKS_CHROMA_PATH=$PROJECT_ROOT/chroma_db
ARABIC_BOOKS_COLLECTION=arabic_books
EOF
    print_status "Created .env file"
else
    print_warning ".env file already exists, not overwriting"
fi

# Update agent .env if it exists
if [ -f "$AGENT_ENV_FILE" ]; then
    print_status "Agent .env file found"
else
    print_warning "Agent .env file not found at $AGENT_ENV_FILE"
    print_info "You may need to configure API keys manually"
fi

# Step 5: Pull Open WebUI image
echo ""
echo "Step 5: Preparing Open WebUI..."
echo "─────────────────────────────────"

print_info "Pulling Open WebUI Docker image..."
docker pull ghcr.io/open-webui/open-webui:main -q 2>/dev/null || \
    docker pull ghcr.io/open-webui/open-webui:main

print_status "Open WebUI image ready"

# Done!
echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Configure API keys in .env (GOOGLE_AI_API_KEY, SERPER_API_KEY)"
echo ""
echo "  2. Start the services:"
echo "     ${BLUE}make run-all${NC}"
echo ""
echo "  3. Open the UI in your browser:"
echo "     ${BLUE}http://localhost:3005${NC}"
echo ""
echo "  4. Select 'dr-tulu' or 'dr-tulu-quick' as your model"
echo ""
echo "Endpoint configured: $ENDPOINT"
echo ""
