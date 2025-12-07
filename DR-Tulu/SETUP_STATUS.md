# DR-Tulu Setup Status for Arch Linux

## ✓ Completed Setup

### Environment
- **OS**: Arch Linux with Hyprland
- **Python**: 3.13.7 (meets requirement >= 3.10) ✓
- **Package Manager**: uv 0.8.22 ✓
- **Virtual Environment**: Created at `agent/.venv/` ✓

### Installation
- **dr-agent-lib**: Installed in development mode ✓
- **All Dependencies**: Installed successfully ✓
- **vLLM Optional**: Ready to install when needed ✓

### Configuration Files Created
1. `.env.example` - Template for API keys
2. `activate.sh` - Quick activation script with API key loading
3. `ARCH_SETUP_GUIDE.md` - Comprehensive setup documentation
4. `SETUP_STATUS.md` - This file

## ⚠ Hardware Limitation

**GPU**: Intel HD Graphics 530 (not CUDA-capable)

**Impact**:
- Cannot run local vLLM servers
- **Solution**: Use OpenAI API instead (recommended for this machine)

## Next Steps - To Get Running

### 1. Get API Keys (5-10 minutes)

These services have free tiers:

1. **Semantic Scholar** (S2_API_KEY)
   - Go to: https://api.semanticscholar.org/
   - Click "Get API Key"
   - Copy the key

2. **Serper** (SERPER_API_KEY)
   - Go to: https://serper.dev/
   - Sign up (free, no credit card)
   - Copy your API key

3. **Jina** (JINA_API_KEY)
   - Go to: https://jina.ai/reader/
   - Sign up and get API key
   - Copy the key

### 2. Create .env File

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Copy the template
cp .env.example .env

# Edit and add your keys
nano .env
```

### 3. Activate Environment and Test

```bash
# Navigate to agent directory
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Option A: Quick one-liner (sets API keys automatically from .env)
source activate.sh

# Option B: Manual activation
source .venv/bin/activate
export $(cat .env | grep -v '^#' | xargs)

# Test import
python -c "import dr_agent; print('✓ dr-agent imported successfully')"
```

### 4. Run Interactive Demo

**Recommended for this machine (uses OpenAI API):**

```bash
# First activate the environment
source activate.sh

# Run the demo
python scripts/launch_chat.py \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "use_browse_agent=false"
```

Or simply:
```bash
source activate.sh
launch_chat
```

## Detailed Instructions

See **`ARCH_SETUP_GUIDE.md`** for:
- Step-by-step installation instructions
- Troubleshooting tips
- Project structure overview
- Best practices for future ML projects
- Available commands and workflows

## Quick Reference

### Activate environment (from any terminal)
```bash
source /home/elwalid/projects/parallax_project/DR-Tulu/agent/activate.sh
```

### View available commands
```bash
ls -la agent/scripts/
```

### Check installed packages
```bash
source .venv/bin/activate
uv pip list
```

### Update packages
```bash
source .venv/bin/activate
uv pip install -e ".[vllm]"  # Add vLLM support if you get a GPU
```

## System Setup Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Python 3.10+ | ✓ | 3.13.7 installed |
| uv package manager | ✓ | 0.8.22 installed |
| Virtual environment | ✓ | `.venv/` created |
| dr-agent-lib | ✓ | Installed in dev mode |
| API keys | ⚠ | Need to be configured |
| GPU for vLLM | ✗ | Intel HD 530 (CPU only) |
| OpenAI API option | ✓ | Available, recommended for this machine |

## For Future GPU Machine

When you have access to an NVIDIA GPU, you can:

1. Install vLLM:
   ```bash
   uv pip install -e ".[vllm]"
   ```

2. Run the full local demo:
   ```bash
   # Terminal 1: MCP Server
   python -m dr_agent.mcp_backend.main --port 8000

   # Terminal 2: vLLM servers
   CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

   # Terminal 3: Launch chat
   python scripts/launch_chat.py --model rl-research/DR-Tulu-8B
   ```

## Resources

- **Paper**: https://allenai.org/papers/drtulu
- **Model Hub**: https://huggingface.co/collections/rl-research/dr-tulu
- **Blog Post**: http://allenai.org/blog/dr-tulu
- **Demo Video**: https://youtu.be/4i0W9qAf8K8
- **Static Demo**: https://dr-tulu.github.io/

## Support

For detailed information:
- See `ARCH_SETUP_GUIDE.md` for setup help
- See `agent/README.md` for workflow documentation
- See `agent/evaluation/README.md` for evaluation details
- See main `README.md` for project overview
