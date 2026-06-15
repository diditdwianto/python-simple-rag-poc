"""RAG query pipeline: question -> retrieve -> grounded prompt -> answer.

Run:  python -m src.query "your question"
"""

import sys

from dotenv import load_dotenv

from src import config
from src.embeddings import embed_query
from src.generate import generate
from src.store import search

NO_INFO = "I don't have enough information to answer that."


def answer(question: str) -> str:
    """Full RAG query: embed question, retrieve, threshold, build prompt, generate."""
    qvec = embed_query(question)
    hits = search(qvec, k=config.TOP_K)

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
    load_dotenv()
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(answer(question))
