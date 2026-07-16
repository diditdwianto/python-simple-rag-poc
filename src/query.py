"""RAG query pipeline: question -> retrieve -> grounded prompt -> answer.

Run:  python -m src.query "your question"
"""

import argparse
import re

from dotenv import load_dotenv

from src import config
from src.embeddings import embed_query
from src.generate import generate
from src.rerank import rerank
from src.store import apply_threshold, fetch_all, search

NO_INFO = "I don't have enough information to answer that."

# Legacy hand-written help file. The catalog is now computed live from the index,
# so this file (if still present) is excluded from the listed topics.
CATALOG_SOURCE = "index.md"

# "What do you know?" style questions are answered from live index metadata, not
# from any stored document. Matched by exact normalized phrase (not substring) so
# a real content question like "what do you know about coffee" is NOT hijacked.
_CATALOG_PHRASES = {
    "help",
    "help me",
    "what can you help me with",
    "what can you help with",
    "what can i ask",
    "what can i ask you",
    "what can you do",
    "what do you know",
    "what knowledge do you have",
    "what knowledge do you have access to",
    "what topics do you have",
    "what topics",
    "what topics are available",
    "what subjects are available",
    "what subjects do you have",
    "list the available topics",
    "list available topics",
    "list topics",
    "list the topics",
    "what is in your knowledge base",
    "whats in your knowledge base",
    "what documents do you have",
    "what files do you have",
    "what data do you have",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (for phrase matching)."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def is_catalog_question(question: str) -> bool:
    """True if the question is a 'what do you know / help' meta-question."""
    return _normalize(question) in _CATALOG_PHRASES


def _title_and_summary(text: str, source: str) -> tuple[str, str]:
    """Derive a display title (first markdown H1) and one-line summary (first
    non-heading line) from a document's opening chunk."""
    title = None
    summary = ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if title is None:
                title = s.lstrip("#").strip()
            continue
        summary = s
        break
    if not title:
        title = source.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
    if len(summary) > 160:
        summary = summary[:159].rstrip() + "…"
    return title, summary


def build_catalog() -> list[dict]:
    """Live list of indexed documents — {source, title, summary} — read straight
    from the index, so it never drifts from what's actually searchable."""
    by_source: dict[str, list[dict]] = {}
    for chunk in fetch_all():
        if chunk["source"] == CATALOG_SOURCE:
            continue
        by_source.setdefault(chunk["source"], []).append(chunk)

    catalog = []
    for source in sorted(by_source):
        opening = min(by_source[source], key=lambda c: c["chunk_index"])["content"]
        title, summary = _title_and_summary(opening, source)
        catalog.append({"source": source, "title": title, "summary": summary})
    return catalog


def format_catalog_answer(catalog: list[dict]) -> str:
    """Render the live catalog as the grounded markdown answer (no LLM call)."""
    if not catalog:
        return "My knowledge base is currently empty — ingest some documents and ask again."
    lines = ["Here's what's in my knowledge base — ask me about any of these:", ""]
    for item in catalog:
        bullet = f"- **{item['title']}** (`{item['source']}`)"
        if item["summary"]:
            bullet += f" — {item['summary']}"
        lines.append(bullet)
    lines += ["", "If your question falls outside these, I'll tell you I don't have enough information to answer."]
    return "\n".join(lines)


def retrieve(question: str, source: str | None = None) -> list[dict]:
    """Two-stage retrieval, then the relevance gate.

    Stage 1: embed the question and search (hybrid or vector). When reranking is
    on we pull RERANK_FETCH_K candidates (a wide recall net) instead of TOP_K.
    Stage 2: the cross-encoder rescores those candidates and keeps the best TOP_K.
    Then apply_threshold gates on the resulting score.

    If `source` is given, retrieval is restricted to chunks from that file.
    Returns the surviving hits (empty list means off-topic / no info).

    NOTE: app.py's /api/query runs this same sequence inline so it can time each
    phase for the pipeline-activity panel — keep the two in sync.
    """
    qvec = embed_query(question)
    fetch_k = config.RERANK_FETCH_K if config.RERANK_ENABLED else config.TOP_K
    hits = search(qvec, k=fetch_k, source=source, query_text=question)
    if config.RERANK_ENABLED:
        hits = rerank(question, hits, top_n=config.TOP_K)
    return apply_threshold(hits)


def build_user_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the grounded user turn: labelled context chunks + the question."""
    context = "\n\n".join(f"[source: {h['source']}]\n{h['content']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str, source: str | None = None) -> str:
    """Full RAG query: retrieve, short-circuit if nothing relevant, else generate."""
    hits = retrieve(question, source=source)
    if not hits:
        return NO_INFO
    return generate(build_user_prompt(question, hits))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question over the ingested docs (RAG).")
    parser.add_argument("question", nargs="?", help="the question to ask")
    parser.add_argument(
        "--source", help="restrict retrieval to a single source file, e.g. rainbowcandy.md"
    )
    args = parser.parse_args()

    load_dotenv()
    question = args.question or input("Question: ")
    print(answer(question, source=args.source))
