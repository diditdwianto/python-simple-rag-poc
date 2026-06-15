"""RAG query pipeline: question -> retrieve -> grounded prompt -> answer.

Run:  python -m src.query "your question"
"""

import argparse

from dotenv import load_dotenv

from src import config
from src.embeddings import embed_query
from src.generate import generate
from src.store import search

NO_INFO = "I don't have enough information to answer that."


def answer(question: str, source: str | None = None) -> str:
    """Full RAG query: embed question, retrieve, threshold, build prompt, generate.

    If `source` is given, retrieval is restricted to chunks from that file.
    """
    qvec = embed_query(question)
    hits = search(qvec, k=config.TOP_K, source=source)

    # Keep only sufficiently-close hits; otherwise don't call the LLM.
    hits = [h for h in hits if h["vector_distance"] <= config.MAX_DISTANCE]
    if not hits:
        return NO_INFO

    context = "\n\n".join(
        f"[source: {h['source']}]\n{h['content']}" for h in hits
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return generate(user_prompt)


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
