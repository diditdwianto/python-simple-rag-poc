"""Raw retrieval — show exactly what Redis returns, with NO LLM step.

Useful for debugging retrieval: you see each hit's distance, source,
chunk_index, and full content, and which ones pass the distance threshold.

Run:  python -m src.query_raw "your question"
"""

import sys

from src import config
from src.embeddings import embed_query
from src.store import search


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")

    qvec = embed_query(question)
    hits = search(qvec, k=config.TOP_K)

    print(f"Query: {question}")
    print(f"Top {config.TOP_K} hits  (MAX_DISTANCE = {config.MAX_DISTANCE}, lower distance = closer)\n")

    if not hits:
        print("(no hits returned)")
        return

    for i, h in enumerate(hits, 1):
        keep = "KEEP" if h["vector_distance"] <= config.MAX_DISTANCE else "drop"
        print(
            f"[{i}] distance={h['vector_distance']:.4f}  [{keep}]  "
            f"source={h['source']}  chunk_index={h['chunk_index']}"
        )
        print(f"    {h['content']}")
        print()


if __name__ == "__main__":
    main()
