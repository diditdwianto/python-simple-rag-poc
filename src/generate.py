"""Pluggable LLM backend. Groq implementation now.

Public surface is `generate(user_prompt) -> str`. A future local backend (Ollama)
would implement the same signature; nothing else in the pipeline changes.
"""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator

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


def generate_stream(user_prompt: str) -> Iterator[tuple[str, object]]:
    """Stream the answer from Groq, yielding ('token', text) per delta.

    Ends with a single ('done', stats) where stats carries the full answer plus
    timing/usage so the UI can show a detailed activity breakdown:
      - first_token: wall seconds until the first content token arrived
      - wall:        total wall seconds for the streamed generation
      - prompt_tokens / completion_tokens: Groq token counts
      - prompt_time / completion_time / queue_time / total_time: Groq server timings

    Groq reports usage in the final stream chunk under `x_groq.usage`; everything
    else here is measured locally.
    """
    start = time.perf_counter()
    first_token_at: float | None = None
    parts: list[str] = []
    usage = None

    stream = _client().chat.completions.create(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        x_groq = getattr(chunk, "x_groq", None)
        if x_groq is not None and getattr(x_groq, "usage", None) is not None:
            usage = x_groq.usage
        elif getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter() - start
                parts.append(delta)
                yield ("token", delta)

    stats: dict[str, object] = {
        "answer": "".join(parts),
        "wall": time.perf_counter() - start,
        "first_token": first_token_at,
    }
    if usage is not None:
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_time",
            "completion_time",
            "queue_time",
            "total_time",
        ):
            stats[field] = getattr(usage, field, None)

    yield ("done", stats)
