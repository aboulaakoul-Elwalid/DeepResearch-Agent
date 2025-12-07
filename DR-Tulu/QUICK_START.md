# DR-Tulu Quick Start (5 minutes to running)

## TL;DR - Get it running now

### 1. Get Free API Keys (Copy-Paste from your browser)

These are completely free with generous limits:

**Option A - Recommended (easiest):**
```bash
# Just open these links and get keys:
# 1. https://serper.dev/ → Sign up → Copy API Key
# 2. https://api.semanticscholar.org/ → Click "Get API Key"
# 3. https://jina.ai/reader/ → Sign up → Get API key
```

### 2. Add Your Keys

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Create your .env file
cat > .env << 'EOF'
S2_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
JINA_API_KEY=your_key_here
EOF
```

### 3. Run It

```bash
# This does everything: activates env, loads keys, runs demo
source activate.sh
python scripts/launch_chat.py
```

**That's it!** Start asking research questions.

---

## Detailed Walkthrough

### Step 1: Get API Keys (3 free services)

#### Service 1: Semantic Scholar (Academic Papers)
```
1. Visit: https://api.semanticscholar.org/
2. Scroll down and click "Get API Key"
3. You'll get a key instantly
4. Copy this key
```

#### Service 2: Serper (Web Search)
```
1. Visit: https://serper.dev/
2. Click "Sign up"
3. Complete form (takes 30 seconds)
4. Go to Dashboard
5. Copy your API Key
6. Free tier: 100 searches/month (plenty for testing)
```

#### Service 3: Jina (Web Reader)
```
1. Visit: https://jina.ai/reader/
2. Click "Sign up"
3. After account creation, go to API settings
4. Generate and copy your API key
```

### Step 2: Setup (One-time)

```bash
# Go to the agent directory
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Create .env file with your keys
cat > .env << 'EOF'
# Paste your actual keys below (no quotes needed)
S2_API_KEY=your_semantic_scholar_key
SERPER_API_KEY=your_serper_key
JINA_API_KEY=your_jina_key
EOF

# Verify it was created
cat .env
```

### Step 3: Run (Every time you want to use it)

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent

# Activate everything in one command
source activate.sh

# Start the demo
python scripts/launch_chat.py
```

Or just run both in one line:
```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent && source activate.sh && python scripts/launch_chat.py
```

## What You Can Do

Once running, you can ask questions like:

```
"What are the latest developments in quantum computing?"
"How does transformer architecture work?"
"What are the best practices for distributed training?"
"Find recent papers on neural architecture search"
"Research the state of explainable AI in 2024"
```

The system will:
1. Search the web and academic databases
2. Read and process the relevant content
3. Synthesize an answer with citations
4. Show you the research process in real-time

## Troubleshooting

### "Module not found" error
```bash
# Reactivate and reinstall
source .venv/bin/activate
uv pip install -e .
```

### "API key rejected"
```bash
# Check your .env file
cat .env

# Make sure keys have no quotes or extra spaces
# Keys should look like: SERPER_API_KEY=abc123def456
# NOT: SERPER_API_KEY="abc123def456"
```

### "Connection refused"
```bash
# Sometimes the chat script auto-launches services
# Try with skip checks flag
python scripts/launch_chat.py --skip-checks
```

### Demo is slow
```bash
# This is normal on first run - models are loading
# It will get faster on subsequent runs
```

## Next Steps

- **Read more**: See `ARCH_SETUP_GUIDE.md` for detailed information
- **Evaluate**: See `agent/evaluation/README.md` for benchmarks
- **Train**: See `rl/open-instruct/README.md` for training your own
- **Custom workflows**: Check `agent/workflows/` directory

## Commands for Power Users

After activating environment (`source activate.sh`):

```bash
# View available workflows
ls workflows/

# Check script options
python scripts/launch_chat.py --help

# View configuration
cat workflows/auto_search_sft.yaml

# Run non-interactive search
bash scripts/auto_search.sh

# Launch just the MCP server
mcp_server
```

## Pro Tips

1. **Save context**: Ask follow-up questions - the system remembers
2. **Specific queries**: More specific = better results
3. **Eval results**: Check `agent/evaluation/` to run benchmarks
4. **Cache clearing**: `.cache-hostname/` directories can be deleted to clear cache

## Got Stuck?

1. Check `ARCH_SETUP_GUIDE.md` - has troubleshooting section
2. Check `agent/README.md` - official documentation
3. Run `python -c "import dr_agent; print('OK')"` to test import
4. Check `.env` file exists and has correct API keys

---

**Happy researching! 🚀**
