# DR-Tulu: Using Gemini, Groq, or OpenAI APIs

Great question! Instead of struggling with local vLLM on CPU, you can use cloud APIs. The project already supports this through **litellm**.

## Your Best Option: Groq API

**Why Groq?**
- ✅ **FREE** - 30k tokens/day free tier (more than enough for testing)
- ✅ **FASTEST** - Sub-second inference latency
- ✅ **Compatible** - Native OpenAI API format
- ✅ **Open Source** - Uses Mixtral 8x7B model

### Quick Setup (3 minutes)

```bash
# 1. Get free API key
open https://console.groq.com/keys  # Sign up and copy key

# 2. Add to your .env
echo "GROQ_API_KEY=your_key_here" >> ~/.bashrc
source ~/.bashrc

# 3. Test it
cd /home/elwalid/projects/parallax_project/DR-Tulu/agent
source activate.sh
python test_apis.py  # Verify it works

# 4. Run the demo
python scripts/launch_chat.py --config workflows/auto_search_groq.yaml
```

That's it! You're now running DR-Tulu with Groq.

---

## Other Options

### Google Gemini API
- **Cost**: FREE tier (15 requests/minute)
- **Speed**: Good
- **Quality**: Excellent (Google's latest models)
- **Good For**: High-quality responses, research
- **Downside**: Rate limited on free tier

```bash
# Get key: https://ai.google.dev/
echo "GOOGLE_API_KEY=your_key" >> .env
python scripts/launch_chat.py --config workflows/auto_search_gemini.yaml
```

### OpenAI API
- **Cost**: Paid (~$0.005 per research query)
- **Speed**: Good
- **Quality**: Excellent (GPT-4o-mini)
- **Good For**: Production use, best quality
- **Setup**: Get key at https://platform.openai.com/api-keys

```bash
echo "OPENAI_API_KEY=your_key" >> .env
python scripts/launch_chat.py --config workflows/auto_search_openai.yaml
```

---

## Available Configs

All configs are pre-made for you:

```
agent/workflows/
├── auto_search_groq.yaml    ← Recommended
├── auto_search_gemini.yaml
├── auto_search_openai.yaml
├── auto_search_sft.yaml     ← Original (needs vLLM)
└── auto_search_sft-oai.yaml ← OpenAI version of original
```

---

## Testing Your API

Before running the full demo, test your API connection:

```bash
source activate.sh
python test_apis.py
```

This will:
- Check if API keys are in .env ✓
- Test connection to each API ✓
- Show which ones are working ✓
- Give you the right startup command ✓

---

## Why This Works

The DR-Tulu architecture supports any OpenAI-compatible API:

```
Original (needs 2 GPUs):
┌─────────┐
│ vLLM    │ ← Requires NVIDIA GPU
└─────────┘

Your Setup (CPU-friendly):
┌──────────────────┐
│ Cloud API        │ ← Runs in cloud
│ (Groq/Gemini)    │
└──────────────────┘
        ↓
   ┌────────────┐
   │ DR-Tulu    │ ← Runs on your machine
   │ Agent      │    (CPU is fine!)
   └────────────┘
        ↓
   Uses: Web Search, Academic Search, Reading
```

**Key insight**: Only the LLM inference runs in the cloud. Everything else (search, synthesis, tool calling) runs locally.

---

## Performance Expectations

### With Groq
- First token latency: <100ms
- Full response: 2-5 seconds
- Free tier: Plenty for development

### With Gemini
- First token latency: 200-500ms
- Full response: 3-8 seconds
- Rate limit: 15 requests/minute

### With OpenAI
- First token latency: 100-300ms
- Full response: 2-5 seconds
- Cost: ~$0.001-0.01 per query

---

## Common Questions

**Q: Is it slower than local inference?**
A: Actually no! Groq is often faster due to optimization. Original vLLM on CPU would be much slower.

**Q: Will this work for serious research?**
A: Yes! Groq's Mixtral 8x7B is a serious model. For production, add OpenAI as a fallback.

**Q: Can I use multiple APIs?**
A: Yes! You can create a custom config using different APIs for search vs. browse:

```yaml
# Use Groq for speed, OpenAI for quality
search_agent_base_url: "https://api.groq.com/openai/v1"
search_agent_model_name: "mixtral-8x7b-32768"

browse_agent_base_url: "https://api.openai.com/v1"
browse_agent_model_name: "gpt-4o-mini"
```

**Q: How much will this cost?**
A: Groq free tier covers most testing. For OpenAI: ~$0.01-0.05 per research query.

---

## Next Steps

1. **Choose one**: Groq (free, fast) or Gemini (free, limited) or OpenAI (paid, best)

2. **Get API key**:
   - Groq: https://console.groq.com/keys
   - Gemini: https://ai.google.dev/
   - OpenAI: https://platform.openai.com/api-keys

3. **Test it**:
   ```bash
   source activate.sh
   python test_apis.py
   ```

4. **Run it**:
   ```bash
   python scripts/launch_chat.py --config workflows/auto_search_groq.yaml
   ```

That's all! You're running a serious deep research agent now.

---

## Files Created for You

Documentation:
- `API_ALTERNATIVES.md` - Complete API guide
- `API_SETUP_GUIDE.md` - This file

Configuration Files:
- `workflows/auto_search_groq.yaml`
- `workflows/auto_search_gemini.yaml`
- `workflows/auto_search_openai.yaml`

Testing:
- `test_apis.py` - Verify your API setup

---

## Troubleshooting

### API key not recognized
```bash
# Make sure it's in .env
cat .env | grep GROQ_API_KEY

# If missing, add it
echo "GROQ_API_KEY=your_key" >> .env

# Reload environment
source activate.sh
```

### Connection timeout
```bash
# Test direct API connection
python test_apis.py

# Check internet
ping api.groq.com
```

### Model not available
Each API has different models. See `API_ALTERNATIVES.md` for the complete list.

---

## Resources

- **Groq**: https://console.groq.com/
- **Gemini**: https://ai.google.dev/
- **OpenAI**: https://platform.openai.com/
- **litellm**: https://docs.litellm.ai/
- **DR-Tulu Paper**: https://allenai.org/papers/drtulu

---

**Ready to research?** Get your API key and run!
