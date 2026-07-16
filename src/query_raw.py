"""Raw retrieval — show exactly what Redis returns, with NO LLM step.

Useful for debugging retrieval: you see each hit's distance, source,
chunk_index, and full content, and which ones pass the distance threshold.

Run:  python -m src.query_raw "your question"
"""

import argparse

from src import config
from src.embeddings import embed_query
from src.rerank import rerank
from src.store import apply_threshold, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw retrieval (no LLM).")
    parser.add_argument("question", nargs="?", help="the query text")
    parser.add_argument(
        "--source", help="restrict retrieval to a single source file, e.g. rainbowcandy.md"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="skip stage-2 reranking (show raw stage-1 order)"
    )
    args = parser.parse_args()

    question = args.question or input("Question: ")
    qvec = embed_query(question)

    use_rerank = config.RERANK_ENABLED and not args.no_rerank
    fetch_k = config.RERANK_FETCH_K if use_rerank else config.TOP_K
    # Pass query_text so this mirrors the real pipeline (hybrid when SEARCH_MODE=hybrid).
    hits = search(qvec, k=fetch_k, source=args.source, query_text=question)
    if use_rerank:
        hits = rerank(question, hits, top_n=config.TOP_K)

    # Which hits the gate would actually keep (identity match on the hit dicts).
    kept = {id(h) for h in apply_threshold(list(hits))}

    print(f"Query: {question}")
    print(f"Search mode: {config.SEARCH_MODE}")
    print(f"Reranking: {'on (' + config.RERANK_MODEL + ')' if use_rerank else 'off'}")
    if args.source:
        print(f"Filter: source == {args.source}")
    if use_rerank:
        print(f"Top {config.TOP_K} of {fetch_k} candidates  "
              f"(RERANK_MIN_SCORE = {config.RERANK_MIN_SCORE}, higher score = closer)\n")
    else:
        print(f"Top {config.TOP_K} hits  "
              f"(MAX_DISTANCE = {config.MAX_DISTANCE}, lower distance = closer)\n")

    if not hits:
        print("(no hits returned)")
        return

    for i, h in enumerate(hits, 1):
        verdict = "KEEP" if id(h) in kept else "drop"
        if "rerank_score" in h:
            metric = f"rerank={h['rerank_score']:+.3f}"
            dist = h.get("vector_distance")
            metric += f"  (dist={dist:.4f})" if dist is not None else "  (dist=n/a)"
        else:
            dist = h.get("vector_distance")
            metric = f"distance={dist:.4f}" if dist is not None else "distance=n/a"
            if "combined_score" in h:
                metric += f"  score={h['combined_score']:.4f}"
        print(
            f"[{i}] {metric}  [{verdict}]  "
            f"source={h['source']}  chunk_index={h['chunk_index']}"
        )
        print(f"    {h['content']}")
        print()


if __name__ == "__main__":
    main()
