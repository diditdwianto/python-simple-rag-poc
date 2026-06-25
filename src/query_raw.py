"""Raw retrieval — show exactly what Redis returns, with NO LLM step.

Useful for debugging retrieval: you see each hit's distance, source,
chunk_index, and full content, and which ones pass the distance threshold.

Run:  python -m src.query_raw "your question"
"""

import argparse

from src import config
from src.embeddings import embed_query
from src.store import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw retrieval (no LLM).")
    parser.add_argument("question", nargs="?", help="the query text")
    parser.add_argument(
        "--source", help="restrict retrieval to a single source file, e.g. rainbowcandy.md"
    )
    args = parser.parse_args()

    question = args.question or input("Question: ")
    qvec = embed_query(question)
    # Pass query_text so this mirrors the real pipeline (hybrid when SEARCH_MODE=hybrid).
    hits = search(qvec, k=config.TOP_K, source=args.source, query_text=question)

    print(f"Query: {question}")
    print(f"Search mode: {config.SEARCH_MODE}")
    if args.source:
        print(f"Filter: source == {args.source}")
    print(f"Top {config.TOP_K} hits  (MAX_DISTANCE = {config.MAX_DISTANCE}, lower distance = closer)\n")

    if not hits:
        print("(no hits returned)")
        return

    for i, h in enumerate(hits, 1):
        dist = h.get("vector_distance")
        if dist is not None:
            keep = "KEEP" if dist <= config.MAX_DISTANCE else "drop"
            metric = f"distance={dist:.4f}"
        else:
            keep = "text-only"  # found by BM25 alone; no vector distance
            metric = "distance=n/a"
        if "combined_score" in h:
            metric += f"  score={h['combined_score']:.4f}"
        print(
            f"[{i}] {metric}  [{keep}]  "
            f"source={h['source']}  chunk_index={h['chunk_index']}"
        )
        print(f"    {h['content']}")
        print()


if __name__ == "__main__":
    main()
