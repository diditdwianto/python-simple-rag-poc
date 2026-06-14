"""Smoke test for the Groq LLM call.

Loads GROQ_API_KEY from the repo-root .env, sends one prompt to a Gemma model
on Groq, and prints the reply. Run this to confirm your key and the groq SDK
work before building the rest of the RAG pipeline.

Usage:
    pip install groq python-dotenv
    python test-script/llm-call-test.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load the .env at the repo root, regardless of where this script is run from.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

# Groq no longer hosts Gemma. Using a small, fast, stable Llama instruct model.
# To see what's available: Groq(api_key=...).models.list(), or
# https://console.groq.com/docs/models
MODEL = "llama-3.1-8b-instant"


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            f"GROQ_API_KEY not found. Expected it in {ENV_PATH}\n"
            "Get a free key at https://console.groq.com (API Keys -> Create), then add:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )

    client = Groq(api_key=api_key)

    prompt = "Reply with one short sentence confirming you are working."

    print(f"Model:  {MODEL}")
    print(f"Prompt: {prompt}\n")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception as exc:  # surface the real error for debugging
        sys.exit(f"Groq call failed: {exc}")

    print("Response:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
