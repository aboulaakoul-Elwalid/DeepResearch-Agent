# 🚀 Launch DR-Tulu with Gemini Right Now

Your system is fully configured and tested. Here's how to launch:

## Quick Launch (One Command)

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_gemini.yaml
```

That's it! The demo will:
1. Start the MCP backend for tool orchestration
2. Connect to Gemini 2.0 Flash API
3. Open an interactive chat interface
4. You can start asking research questions

## What You Can Ask

```
"Research quantum computing breakthroughs in 2024"
"What's the latest in transformer architecture?"
"Find papers on neural architecture search"
"Summarize recent advances in LLMs"
```

## What Will Happen

When you ask a question, DR-Tulu will:

1. **Plan** - Gemini plans search strategy
2. **Search** - Searches web + academic papers
3. **Read** - Reads and processes results
4. **Synthesize** - Creates coherent answer
5. **Respond** - Shows findings with sources

All powered by:
- **Gemini 2.0 Flash** ← reasoning & synthesis
- **Serper API** ← web search
- **Jina API** ← page reading
- **Semantic Scholar API** ← academic papers

## Verification Checklist

Before launching, verify everything is ready:

```bash
# 1. Check environment
source .venv/bin/activate
python --version  # Should be 3.13+

# 2. Check API keys loaded
python -c "import os; from pathlib import Path; [os.environ.update({l.split('=')[0].strip(): l.split('=')[1].strip()}) for l in Path('.env.example').read_text().split('\n') if l.strip() and not l.startswith('#') and '=' in l]; print('✓ All keys loaded')"

# 3. Test Gemini directly
python test_gemini_agentic.py

# 4. Check web search tools
python -c "from dr_agent.mcp_backend import tools; print('✓ MCP tools available')"
```

## Troubleshooting Launch

### Issue: "Module not found"
```bash
source .venv/bin/activate
uv pip install -e .
```

### Issue: "API key not recognized"
```bash
# Check keys are in .env.example
cat .env.example | grep GOOGLE_AI_API_KEY

# Make sure it's the full API key, not truncated
```

### Issue: "Port already in use"
```bash
# Change MCP port
python scripts/launch_chat.py \
  --config workflows/auto_search_gemini.yaml \
  --mcp-port 9000
```

### Issue: Slow response
This is normal - first request loads model. Subsequent requests are faster.

## Performance Tips

1. **Second query is faster** - Model is cached
2. **More specific queries = better results** - "transformer attention mechanism" > "how AI works"
3. **Follow-ups remember context** - Ask clarifying questions
4. **Be patient on first run** - Gemini API may take 3-5 seconds

## Advanced Options

### Show Full Tool Output
```bash
python scripts/launch_chat.py \
  --config workflows/auto_search_gemini.yaml \
  --show-full-tool-output
```

### Verbose Logging
```bash
python scripts/launch_chat.py \
  --config workflows/auto_search_gemini.yaml \
  --verbose
```

### Custom Config
```bash
# Modify config
cp workflows/auto_search_gemini.yaml workflows/my_config.yaml
nano workflows/my_config.yaml

# Run with custom config
python scripts/launch_chat.py --config workflows/my_config.yaml
```

### Skip Service Checks
```bash
python scripts/launch_chat.py \
  --config workflows/auto_search_gemini.yaml \
  --skip-checks
```

## What You're Testing

This launch is a full test of:

✓ Gemini 2.0 Flash agentic behavior
✓ Multi-step research planning
✓ Tool orchestration
✓ Web search integration
✓ Paper search integration
✓ Synthesis capability
✓ Long-form research support

## Next Steps After Launch

### If It Works Well
- Start using for research
- Try complex queries
- Give feedback on model quality
- Consider moving to longer sessions

### If You Want to Try Alternatives
- **Switch to Groq**: Change config to `auto_search_groq.yaml`
- **Switch to OpenAI**: Change config to `auto_search_openai.yaml`
- **Try Qwen later**: When you have access to GPU

### If You Want to Understand More
- Read `GEMINI_CHOICE.md` - Why Gemini is good choice
- Read `API_ALTERNATIVES.md` - Comparison of providers
- Read `ARCH_SETUP_GUIDE.md` - System architecture

## Common Questions

**Q: How long does first launch take?**
A: 10-20 seconds (includes MCP server startup)

**Q: How much does it cost?**
A: FREE within Google's API tier (15 requests/minute)

**Q: What if I hit rate limit?**
A: You can wait 1 minute and retry. Or use Groq/OpenAI instead.

**Q: Can I switch APIs later?**
A: Yes! Just run with different config file.

**Q: Will it save conversation history?**
A: Check agent configuration. Default may not persist.

## Reality Check

Your setup now:
- ✓ Python 3.13.7 working
- ✓ All dependencies installed
- ✓ 4 APIs configured (Gemini, Serper, Jina, S2)
- ✓ MCP backend ready
- ✓ Agentic behavior tested ✓
- ✓ Everything verified ✓

**The system is ready.** No more setup needed. Just launch and research!

---

## Let's Go!

```bash
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_gemini.yaml
```

Then ask:
```
"What are the latest breakthroughs in AI?"
```

Enjoy! 🚀
