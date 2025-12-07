# DR-Tulu Setup Guide for Arch Linux (Hyprland)

This guide provides a step-by-step setup for running DR-Tulu on Arch Linux without conda.

## System Information

- **OS**: Arch Linux
- **GPU**: Intel HD Graphics 530 (Note: vLLM requires NVIDIA/AMD GPUs)
- **Python**: 3.13.7
- **Package Manager**: uv (recommended) + venv

## Prerequisites

### 1. Install Required System Dependencies

```bash
# On Arch Linux, you should already have:
# - Python 3.10+
# - pip/uv

# For optional features (crawl4ai with playwright):
sudo pacman -S chromium  # or firefox
```

### 2. Check uv Installation

```bash
uv --version
# Should show: uv 0.x.x
```

If not installed:
```bash
pip install uv
```

## Setup Steps

### Step 1: Create Python Virtual Environment

Instead of conda, use Python venv:

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify Python version (should be >= 3.10)
python --version
```

### Step 2: Install DR-Agent Library

```bash
# Install in development mode with all dependencies
uv pip install -e .

# Or install from PyPI (once released)
uv pip install dr_agent
```

### Step 3: Configure API Keys

Create a `.env` file in the agent directory:

```bash
cp .env.example .env
nano .env  # or vim, your favorite editor
```

Fill in your API keys:

**API Key Sources (all have free tiers):**

1. **Semantic Scholar API** (S2_API_KEY)
   - Visit: https://api.semanticscholar.org/
   - Click "Get API key"
   - Free tier: good for research

2. **Serper API** (SERPER_API_KEY)
   - Visit: https://serper.dev/
   - Sign up for free account
   - Free tier: 100 searches/month

3. **Jina API** (JINA_API_KEY)
   - Visit: https://jina.ai/reader/
   - Sign up and get API key
   - Free tier: available

4. **OpenAI API Key** (Optional - OPENAI_API_KEY)
   - For running with ChatGPT instead of local models
   - Visit: https://platform.openai.com/api-keys
   - Requires payment, but very flexible

### Step 4: Run the Interactive Demo

#### Option A: Using OpenAI API (Recommended for CPU-only systems)

```bash
# Activate venv
source .venv/bin/activate

# Set OpenAI key
export OPENAI_API_KEY="your_key_here"

# Export other API keys
export SERPER_API_KEY="your_key_here"
export S2_API_KEY="your_key_here"
export JINA_API_KEY="your_key_here"

# Run the demo
uv run --extra vllm python scripts/launch_chat.py \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "use_browse_agent=false"
```

#### Option B: Using Local vLLM (Requires NVIDIA/AMD GPU)

```bash
# On a machine with CUDA-capable GPU:

# Terminal 1 - Launch MCP server
MCP_CACHE_DIR=".cache-$(hostname)" python -m dr_agent.mcp_backend.main --port 8000

# Terminal 2 - Launch vLLM servers
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

# Terminal 3 - Run the demo
uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B
```

## Project Structure

```
DR-Tulu/
├── agent/              # Main agent library (dr-agent-lib)
│   ├── .venv/         # Virtual environment (created)
│   ├── pyproject.toml # Package configuration
│   ├── scripts/       # Executable scripts
│   ├── dr_agent/      # Source code
│   └── workflows/     # Configuration files
├── rl/                # RL training code (Open-Instruct based)
├── sft/               # SFT training code (LLaMA-Factory based)
└── README.md          # Main documentation
```

## Useful Commands

### Activate environment
```bash
source /home/elwalid/projects/parallax_project/DR-Tulu/agent/.venv/bin/activate
```

### Check installed packages
```bash
uv pip freeze
```

### View available scripts
```bash
ls agent/scripts/
```

### Run different workflows
```bash
# Auto-search workflow
bash agent/scripts/auto_search.sh

# Interactive chat
uv run python agent/scripts/launch_chat.py --help
```

## Troubleshooting

### Issue: "No module named 'vllm'"
**Solution**: Install vllm extras
```bash
uv pip install -e ".[vllm]"
```

### Issue: "CUDA not available" when using vLLM
**Solution**: This system has Intel HD Graphics 530. You need an NVIDIA GPU.
Use OpenAI API instead (see Option A above).

### Issue: API key not recognized
**Solution**: Make sure `.env` file is in the agent directory and sourced:
```bash
export $(cat .env | xargs)
```

## Next Steps

1. **Get API Keys**: Visit the links above to get free API keys
2. **Update .env**: Fill in your .env file
3. **Test the Demo**: Run the interactive chat
4. **Explore Workflows**: Check `workflows/` directory for different configurations
5. **Review Evaluation**: See `agent/evaluation/README.md` for benchmark info

## Resources

- **Paper**: https://allenai.org/papers/drtulu
- **Models**: https://huggingface.co/collections/rl-research/dr-tulu
- **Blog**: http://allenai.org/blog/dr-tulu
- **Video**: https://youtu.be/4i0W9qAf8K8
- **Static Demo**: https://dr-tulu.github.io/

## Notes for Future Projects

### Best Practices for ML/Research Projects on Arch:

1. **Use venv instead of conda** - lighter weight, no dependency conflicts
2. **Use uv for package management** - faster than pip
3. **Create .env files for API keys** - never commit credentials
4. **Document system requirements** - especially GPU needs
5. **Use git** for version control - included in this repo
6. **Keep virtual environments in project directories** - makes cleanup easy

### Useful Arch Packages for ML/Research:

```bash
# Python development
sudo pacman -S python python-pip python-numpy python-scipy python-scikit-learn

# CUDA/cuDNN (if you get NVIDIA GPU)
sudo pacman -S cuda cudnn

# Development tools
sudo pacman -S base-devel git

# Browser automation (optional)
sudo pacman -S chromium selenium
```

## Getting Help

- Check the official README files in each subdirectory
- See `agent/evaluation/README.md` for evaluation instructions
- GitHub issues: https://github.com/allenai/open-instruct (base project)
