# Using Alternative APIs with DR-Tulu

Since your system doesn't have a GPU for local vLLM inference, you can easily use cloud APIs. This guide shows 4 options (including your Parallax Modal endpoint):

## Quick Comparison

| Provider | Cost | Speed | Quality | Free Tier |
|----------|------|-------|---------|-----------|
| **Groq** | $0.05/1M tokens | ⚡⚡⚡ Fastest | Good | 30k tokens/day |
| **Gemini** | $0.075/1M input | ⚡⚡ Medium | Excellent | 15 req/min free |
| **OpenAI** | $0.15/1M tokens | ⚡⚡ Medium | Excellent | Pay-as-you-go |
| **Parallax (Modal)** | Your Modal GPU time | ⚡⚡ Medium | TBD (Qwen 0.5B) | N/A (private) |

**Recommendation: Use Groq** - It's FREE for moderate use and MUCH faster.

---

## Option 1: Groq API (Recommended - Free & Fast)

### Setup (2 minutes)

1. **Get API Key**
   - Visit: https://console.groq.com/keys
   - Sign up (free)
   - Copy your API key

2. **Add to .env**
   ```bash
   echo "GROQ_API_KEY=your_key_here" >> .env
   ```

3. **Run**
   ```bash
   source activate.sh
   python scripts/launch_chat.py \
     --config workflows/auto_search_groq.yaml \
     --config-overrides "search_agent_api_key=$GROQ_API_KEY"
   ```

### Why Groq?
- ✅ FREE tier: 30k tokens/day (more than enough for testing)
- ✅ FASTEST: Sub-second latency on inference
- ✅ OpenAI compatible: Works drop-in with existing code
- ✅ High quality: Uses Mixtral 8x7B model

### Available Models
```
mixtral-8x7b-32768    # Balanced, great for reasoning
gemma-7b-it           # Smaller, faster
llama-3.1-8b-instant  # Fast, good quality
```

---

## Option 2: Google Gemini API

### Setup (2 minutes)

1. **Get API Key**
   - Visit: https://ai.google.dev/
   - Click "Get API Key"
   - Copy the key

2. **Add to .env**
   ```bash
   echo "GOOGLE_API_KEY=your_key_here" >> .env
   ```

3. **Run with litellm wrapper**
   ```bash
   source activate.sh

   # Method 1: Using litellm (handles auth automatically)
   GOOGLE_API_KEY=$GOOGLE_API_KEY python scripts/launch_chat.py \
     --config workflows/auto_search_gemini.yaml

   # Method 2: Direct litellm call
   python -c "
   from litellm import completion
   response = completion(
       model='gemini-2.0-flash',
       messages=[{'role': 'user', 'content': 'Hello'}],
   )
   print(response)
   "
   ```

### Why Gemini?
- ✅ FREE tier: 15 requests/minute
- ✅ Very good quality: Google's latest models
- ✅ Well integrated with litellm (already in dependencies)
- ❌ Rate limited on free tier (okay for demo, not production)

### Available Models
```
gemini-2.0-flash      # Latest, fast
gemini-1.5-flash      # Proven, stable
gemini-1.5-pro        # Most capable
```

---

## Option 3: OpenAI API

### Setup (2 minutes)

1. **Get API Key**
   - Visit: https://platform.openai.com/api-keys
   - Create new key
   - Copy it

2. **Add to .env**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" >> .env
   ```

3. **Run**
   ```bash
   source activate.sh
   python scripts/launch_chat.py \
     --config workflows/auto_search_openai.yaml \
     --config-overrides "search_agent_api_key=$OPENAI_API_KEY"
   ```

### Why OpenAI?
- ✅ Industry standard
- ✅ Excellent quality
- ✅ No free tier, but cheap: ~$0.005 per request
- ✅ Works with existing litellm code

### Models
```
gpt-4o-mini           # Fast, cheap, good quality
gpt-4-turbo          # More capable
gpt-4o               # Newest, fastest
```

---

## Option 4: Parallax (Modal, internal endpoint)

### Setup (already deployed)

- Endpoint: `https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Auth: None required (pass any dummy key)

### Run

```bash
source activate.sh
python scripts/launch_chat.py \
  --config workflows/auto_search_parallax.yaml \
  --skip-checks  # optional: skip local vLLM health probe
```

### Failover

The workflow includes an automatic failover: if the Parallax endpoint is unreachable, it switches to Gemini (`gemini/gemini-2.0-flash`) using your existing `GOOGLE_AI_API_KEY`, so you can continue testing the UI without changing configs.

### Notes
- Uses the OpenAI-compatible Parallax endpoint you deployed on Modal.
- Set `ARABIC_BOOKS_CHROMA_PATH`/`ARABIC_BOOKS_COLLECTION` if you want the `search_arabic_books` tool active.
- Keep `--skip-checks` if the launcher warns about `/health` on the remote endpoint.

---

## How to Use Each Config

All four options are ready to use. Just choose one:

### Groq (Fastest & Free)
```bash
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_groq.yaml
```

### Gemini (Google's Models)
```bash
export GOOGLE_API_KEY="your_key"
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_gemini.yaml
```

### OpenAI (Industry Standard)
```bash
export OPENAI_API_KEY="your_key"
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_openai.yaml
```

### Parallax (Modal GPU)
```bash
source activate.sh
python scripts/launch_chat.py --config workflows/auto_search_parallax.yaml --skip-checks
```

---

## Customizing Configs

You can also create your own config by copying an existing one:

```bash
cp workflows/auto_search_groq.yaml workflows/auto_search_custom.yaml
nano workflows/auto_search_custom.yaml
```

Edit these key fields:
- `search_agent_base_url` - API endpoint
- `search_agent_model_name` - Model to use
- `search_agent_api_key` - Your API key

---

## Testing Your Setup

Before running the full demo, test the connection:

### Test Groq
```bash
source activate.sh
python -c "
from litellm import completion
import os

response = completion(
    model='groq/mixtral-8x7b-32768',
    api_key=os.getenv('GROQ_API_KEY'),
    messages=[{'role': 'user', 'content': 'Hello! Just a quick test.'}],
)
print('✓ Groq working!')
print(response['choices'][0]['message']['content'][:100])
"
```

### Test Gemini
```bash
source activate.sh
python -c "
from litellm import completion
import os

response = completion(
    model='gemini-2.0-flash',
    api_key=os.getenv('GOOGLE_API_KEY'),
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print('✓ Gemini working!')
print(response['choices'][0]['message']['content'][:100])
"
```

### Test OpenAI
```bash
source activate.sh
python -c "
from litellm import completion
import os

response = completion(
    model='gpt-4o-mini',
    api_key=os.getenv('OPENAI_API_KEY'),
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print('✓ OpenAI working!')
print(response['choices'][0]['message']['content'][:100])
"
```

---

## Common Issues

### "API key not found"
Make sure your API key is in `.env`:
```bash
cat .env | grep GROQ_API_KEY  # Check it's there
source activate.sh            # Reload
```

### "Rate limited"
If using Gemini free tier, it has 15 req/min limit.
Solution: Wait a minute or upgrade to paid.

### "Model not found"
Check available models for your provider:
- Groq: https://console.groq.com/docs/models
- Gemini: https://ai.google.dev/models
- OpenAI: https://platform.openai.com/docs/models

### "Connection timeout"
Check your internet connection:
```bash
ping api.groq.com
curl https://api.groq.com/health
```

---

## Cost Estimates

For a typical research session (10 queries × 2000 tokens each):

| Provider | Cost |
|----------|------|
| Groq | $0 (within free tier) |
| Gemini | $0 (within free tier) |
| OpenAI | ~$0.03 |

---

## Next Steps

1. Choose your preferred API above
2. Get the API key
3. Add to .env
4. Run the appropriate config
5. Start researching!

---

## Advanced: Using Multiple Models

You can mix different providers in one config:

```yaml
# Use Groq for search (fast), Gemini for browse (good quality)
search_agent_base_url: "https://api.groq.com/openai/v1"
search_agent_model_name: "mixtral-8x7b-32768"

browse_agent_base_url: "https://api.openai.com/v1"  # Different provider!
browse_agent_model_name: "gpt-4o-mini"
```

This can optimize for both speed and quality.

---

## Resources

- **Groq**: https://console.groq.com/
- **Gemini**: https://ai.google.dev/
- **OpenAI**: https://platform.openai.com/
- **litellm Docs**: https://docs.litellm.ai/
