"""RAG query pipeline: question -> retrieve -> grounded prompt -> answer.

Run:  python -m src.query "your question"
"""

import argparse

from dotenv import load_dotenv

from src import config
from src.embeddings import embed_query
from src.generate import generate
from src.store import apply_threshold, search

NO_INFO = "I don't have enough information to answer that."


def retrieve(question: str, source: str | None = None) -> list[dict]:
    """Embed the question, search (hybrid or vector), and apply the relevance gate.

    If `source` is given, retrieval is restricted to chunks from that file.
    Returns the surviving hits (empty list means off-topic / no info).
    """
    qvec = embed_query(question)
    hits = search(qvec, k=config.TOP_K, source=source, query_text=question)
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
