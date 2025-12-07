#!/usr/bin/env python3
"""Quick test script to verify API keys and connectivity."""

import os
import sys
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded .env from {env_file}")
    else:
        print(f"⚠ No .env file found at {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed, using environment variables only")

try:
    from litellm import completion
    print("✓ litellm imported successfully")
except ImportError:
    print("✗ litellm not found. Install with: uv pip install litellm")
    sys.exit(1)

def test_groq():
    """Test Groq API."""
    print("\n" + "="*60)
    print("Testing Groq API...")
    print("="*60)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠ GROQ_API_KEY not set. Get one at: https://console.groq.com/keys")
        return False

    try:
        print("Sending test request to Groq (mixtral-8x7b-32768)...")
        response = completion(
            model="groq/mixtral-8x7b-32768",
            api_key=api_key,
            messages=[{
                "role": "user",
                "content": "Say 'Hello from Groq!' and nothing else."
            }],
            max_tokens=50,
        )

        result = response["choices"][0]["message"]["content"]
        print(f"✓ Groq Response: {result[:100]}")
        print("✓ Groq API working!")
        return True

    except Exception as e:
        print(f"✗ Groq API error: {str(e)[:200]}")
        return False

def test_gemini():
    """Test Google Gemini API."""
    print("\n" + "="*60)
    print("Testing Google Gemini API...")
    print("="*60)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ GOOGLE_API_KEY not set. Get one at: https://ai.google.dev/")
        return False

    try:
        print("Sending test request to Gemini (gemini-2.0-flash)...")
        response = completion(
            model="gemini-2.0-flash",
            api_key=api_key,
            messages=[{
                "role": "user",
                "content": "Say 'Hello from Gemini!' and nothing else."
            }],
            max_tokens=50,
        )

        result = response["choices"][0]["message"]["content"]
        print(f"✓ Gemini Response: {result[:100]}")
        print("✓ Gemini API working!")
        return True

    except Exception as e:
        print(f"✗ Gemini API error: {str(e)[:200]}")
        return False

def test_openai():
    """Test OpenAI API."""
    print("\n" + "="*60)
    print("Testing OpenAI API...")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set. Get one at: https://platform.openai.com/api-keys")
        return False

    try:
        print("Sending test request to OpenAI (gpt-4o-mini)...")
        response = completion(
            model="gpt-4o-mini",
            api_key=api_key,
            messages=[{
                "role": "user",
                "content": "Say 'Hello from OpenAI!' and nothing else."
            }],
            max_tokens=50,
        )

        result = response["choices"][0]["message"]["content"]
        print(f"✓ OpenAI Response: {result[:100]}")
        print("✓ OpenAI API working!")
        return True

    except Exception as e:
        print(f"✗ OpenAI API error: {str(e)[:200]}")
        return False

def main():
    print("\n" + "="*60)
    print("DR-Tulu API Configuration Test")
    print("="*60)

    # Check .env file
    print("\nEnvironment Variables Status:")
    print(f"  GROQ_API_KEY: {'✓ Set' if os.getenv('GROQ_API_KEY') else '✗ Not set'}")
    print(f"  GOOGLE_API_KEY: {'✓ Set' if os.getenv('GOOGLE_API_KEY') else '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Not set'}")
    print(f"  SERPER_API_KEY: {'✓ Set' if os.getenv('SERPER_API_KEY') else '✗ Not set'}")
    print(f"  JINA_API_KEY: {'✓ Set' if os.getenv('JINA_API_KEY') else '✗ Not set'}")

    # Test APIs
    results = {
        "Groq": test_groq(),
        "Gemini": test_gemini(),
        "OpenAI": test_openai(),
    }

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    working = [name for name, result in results.items() if result]
    if working:
        print(f"\n✓ Working APIs: {', '.join(working)}")
        print(f"\nNext steps:")
        print(f"1. Run: source activate.sh")
        if "Groq" in working:
            print(f"2. python scripts/launch_chat.py --config workflows/auto_search_groq.yaml")
        elif "Gemini" in working:
            print(f"2. python scripts/launch_chat.py --config workflows/auto_search_gemini.yaml")
        elif "OpenAI" in working:
            print(f"2. python scripts/launch_chat.py --config workflows/auto_search_openai.yaml")
    else:
        print("\n✗ No working APIs found. Please:")
        print("1. Get API keys from:")
        print("   - Groq: https://console.groq.com/keys (FREE)")
        print("   - Gemini: https://ai.google.dev/ (FREE)")
        print("   - OpenAI: https://platform.openai.com/api-keys (Paid)")
        print("2. Add them to .env:")
        print("   echo 'GROQ_API_KEY=your_key' >> .env")
        print("3. Run this script again to test")
        sys.exit(1)

if __name__ == "__main__":
    main()
