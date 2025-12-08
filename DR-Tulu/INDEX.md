# DR-Tulu Documentation Index

Complete guide to all documentation created for your Arch Linux setup.

## 🚀 Start Here

**New to the project?** Read these in order:

1. **[QUICK_START.md](QUICK_START.md)** ← Start with this (5 minutes)
   - Get API keys
   - Create .env file
   - Run the demo

2. **[API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)** ← Choose your API
   - Groq (recommended, free, fast)
   - Gemini (free, limited)
   - OpenAI (paid, best quality)

3. **[agent/API_ALTERNATIVES.md](agent/API_ALTERNATIVES.md)** ← Detailed comparison
   - Cost analysis
   - Performance benchmarks
   - Setup instructions for each

## 📚 Detailed Guides

### Environment Setup
- **[ARCH_SETUP_GUIDE.md](ARCH_SETUP_GUIDE.md)** - Complete Linux/Hyprland setup
  - System requirements
  - Installation steps
  - Troubleshooting
  - Best practices for future projects

### System Status
- **[SETUP_STATUS.md](SETUP_STATUS.md)** - Current system state
  - What's installed
  - What's missing
  - Hardware limitations
  - Next steps

- **[SETUP_COMPLETE.txt](SETUP_COMPLETE.txt)** - Completion checklist
  - What was done
  - Files created
  - Quick commands

### Project Information
- **[README.md](README.md)** - Original project documentation
  - Paper and blog links
  - Project overview
  - Links to models and datasets

## 🔧 Configuration Files

### Ready-to-Use API Configs

Located in `agent/workflows/`:

- **[auto_search_groq.yaml](agent/workflows/auto_search_groq.yaml)** ← Use this
  - Groq API configuration (FREE, FAST)
  - Pre-configured and ready

- **[auto_search_gemini.yaml](agent/workflows/auto_search_gemini.yaml)**
  - Google Gemini configuration
  - Alternative option

- **[auto_search_parallax.yaml](agent/workflows/auto_search_parallax.yaml)**
  - Local Parallax endpoint (Qwen 7B, OpenAI-compatible)
  - Requires local GPU running Parallax on port 3001

- **[auto_search_openai.yaml](agent/workflows/auto_search_openai.yaml)**
  - OpenAI API configuration
  - Most capable option

- **[auto_search_sft.yaml](agent/workflows/auto_search_sft.yaml)** (original)
  - Requires local vLLM servers
  - For GPU machines only

### Environment Files

In `agent/`:

- **[.env.example](agent/.env.example)** - Template for API keys
  - Copy and fill with your keys
  - Never commit the actual .env file

## 🛠️ Helper Scripts

In `agent/`:

- **[activate.sh](agent/activate.sh)** - Quick activation script
  - Activates venv
  - Loads .env file
  - Provides convenience functions
  - Usage: `source activate.sh`

- **[test_apis.py](agent/test_apis.py)** - Verify API setup
  - Tests Groq, Gemini, OpenAI
  - Checks API keys
  - Verifies connectivity
  - Usage: `python test_apis.py`

## 📋 Quick Reference

### Environment Structure
```
DR-Tulu/
├── agent/                          # Main agent library
│   ├── .venv/                      # Python virtual environment (created)
│   ├── .env.example                # API key template
│   ├── activate.sh                 # Quick activation script
│   ├── test_apis.py                # API verification script
│   ├── API_ALTERNATIVES.md         # API comparison guide
│   ├── scripts/
│   │   └── launch_chat.py          # Main CLI demo launcher
│   ├── workflows/
│   │   ├── auto_search_groq.yaml   # Use this (Groq)
│   │   ├── auto_search_gemini.yaml # Alternative (Gemini)
│   │   ├── auto_search_parallax.yaml # Local Parallax (OpenAI-compatible)
│   │   └── auto_search_openai.yaml # Alternative (OpenAI)
│   └── dr_agent/                   # Source code
├── rl/                             # RL training code
├── sft/                            # SFT training code
├── QUICK_START.md                  # 5-minute setup
├── ARCH_SETUP_GUIDE.md             # Detailed Linux setup
├── API_SETUP_GUIDE.md              # API configuration guide
├── SETUP_STATUS.md                 # Current system status
├── SETUP_COMPLETE.txt              # Completion checklist
├── INDEX.md                        # This file
└── README.md                       # Original documentation
```

## 🎯 Common Tasks

### I want to run the demo right now
```bash
# 1. Get free API key
open https://console.groq.com/keys

# 2. Add to environment
echo "GROQ_API_KEY=your_key" >> ~/.bashrc && source ~/.bashrc

# 3. Run
cd agent && source activate.sh && python test_apis.py
python scripts/launch_chat.py --config workflows/auto_search_groq.yaml
```

### I want to understand the setup
→ Read **[ARCH_SETUP_GUIDE.md](ARCH_SETUP_GUIDE.md)**

### I want to compare API options
→ Read **[API_ALTERNATIVES.md](agent/API_ALTERNATIVES.md)**

### I want to troubleshoot an issue
→ See troubleshooting sections in:
- ARCH_SETUP_GUIDE.md
- API_ALTERNATIVES.md
- API_SETUP_GUIDE.md

### I want to use a different API provider
→ Follow instructions in **[API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)**

### I got a GPU and want to run locally
→ See "GPU Setup Instructions" in **[ARCH_SETUP_GUIDE.md](ARCH_SETUP_GUIDE.md)**

## 📊 System Information

Your System:
- **OS**: Arch Linux with Hyprland
- **Python**: 3.13.7
- **GPU**: Intel HD Graphics 530 (CPU-only)
- **Status**: ✓ Ready for cloud API inference

Environment:
- **Python venv**: ✓ Created and configured
- **Dependencies**: ✓ All installed
- **API Configs**: ✓ Ready for Groq, Gemini, OpenAI
- **Tools**: ✓ Web search, Academic search ready

## 🔗 External Resources

### API Providers
- **Groq**: https://console.groq.com/keys (FREE)
- **Google Gemini**: https://ai.google.dev/ (FREE tier)
- **OpenAI**: https://platform.openai.com/api-keys (Paid)

### Search APIs (Already configured)
- **Serper**: https://serper.dev/ (web search)
- **Jina**: https://jina.ai/reader/ (web reading)
- **Semantic Scholar**: https://api.semanticscholar.org/ (academic papers)

### Documentation
- **DR-Tulu Paper**: https://allenai.org/papers/drtulu
- **Blog Post**: http://allenai.org/blog/dr-tulu
- **Video Demo**: https://youtu.be/4i0W9qAf8K8
- **Model Hub**: https://huggingface.co/collections/rl-research/dr-tulu

### Technical Docs
- **litellm**: https://docs.litellm.ai/
- **vLLM**: https://docs.vllm.ai/
- **MCP (Claude)**: https://modelcontextprotocol.io/

## ✅ Checklist

- [x] Python environment set up
- [x] All dependencies installed
- [x] API configuration files created
- [x] Helper scripts ready
- [x] Complete documentation generated
- [ ] Get Groq API key
- [ ] Create .env file
- [ ] Run test_apis.py
- [ ] Launch demo

## 💡 Tips

1. **Always activate environment first**
   ```bash
   source activate.sh
   ```

2. **Test APIs before running demo**
   ```bash
   python test_apis.py
   ```

3. **Keep API keys secure**
   - Never commit .env file to git
   - Use separate keys for dev/prod
   - Rotate keys regularly

4. **Monitor costs** (if using OpenAI)
   - Watch your usage dashboard
   - Set spending limits
   - Use cheaper models for dev

5. **For best results**
   - Be specific in research queries
   - Follow up with clarifications
   - Check citations and sources

## 📞 Support

For issues:
1. Check relevant documentation file
2. Run `python test_apis.py` to verify setup
3. Check log files in `/tmp/`
4. Review troubleshooting sections in docs

## 🎓 Learning Resources

- Start with **QUICK_START.md** (5 min)
- Read **API_SETUP_GUIDE.md** (10 min)
- Explore **ARCH_SETUP_GUIDE.md** (15 min)
- Review **API_ALTERNATIVES.md** for deep dive (20 min)

**Total time to understand full setup: ~50 minutes**
**Time to get running: ~5 minutes**

---

**Happy researching!** 🚀

For more information on the original project, see [README.md](README.md).
