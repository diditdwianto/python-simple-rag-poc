"""Pluggable LLM backend. Groq implementation now.

Public surface is `generate(user_prompt) -> str`. A future local backend (Ollama)
would implement the same signature; nothing else in the pipeline changes.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src import config

# Load the repo-root .env at import, regardless of caller or working directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using ONLY "
    "the information in the provided context. Rules:\n"
    "- If the answer is not in the context, reply exactly: "
    '"I don\'t have enough information to answer that." Do not use outside knowledge.\n'
    "- Do not guess or assume.\n"
    "- Cite the [source] label(s) you used after the relevant sentence.\n"
    "- Be concise and answer directly."
)


@lru_cache(maxsize=1)
def _client() -> Groq:
    """Create the Groq client lazily (so import never fails on a missing key)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — check your .env at the repo root.")
    return Groq(api_key=api_key)


def generate(user_prompt: str) -> str:
    """Send SYSTEM_PROMPT + user_prompt to the Groq model, return the text answer."""
    resp = _client().chat.completions.create(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.TEMPERATURE,
    )
    return resp.choices[0].message.content or ""
