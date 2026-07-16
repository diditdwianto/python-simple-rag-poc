"""Cross-encoder reranking: the precision half of a two-stage retriever.

Stage 1 (store.search) is a bi-encoder + BM25: it embeds the question and each
chunk *independently* and compares vectors. Fast — every chunk vector is
precomputed at ingest — but it never sees the question and the chunk together,
so it can't judge how they actually relate. It casts a wide, cheap net.

Stage 2 (here) is a cross-encoder: it feeds the (question, chunk) pair through
one model as a single joint input and reads them together, producing a far more
accurate relevance score. Too slow to run over the whole corpus, so we only ever
rerank the handful of candidates stage 1 already surfaced.

Cast wide with the bi-encoder, then read closely with the cross-encoder.

Scores are raw logits (higher = more relevant), NOT cosine distances and NOT
0..1 — the relevance gate keys on them differently from vector distance. See
store.apply_threshold and the RERANK_* settings in config.
"""

from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def _model():
    """Load the cross-encoder once, lazily.

    Lazy + cached so importing this module is free and a run with
    RERANK_ENABLED=False never pays the model-load cost. Imported inside the
    function so `sentence_transformers` is pulled in only when reranking is used.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def warmup() -> None:
    """Eagerly load the cross-encoder now (e.g. at web-server startup).

    Otherwise the very first query pays the one-time model load (~10 s) *inside*
    its own "Reranking" phase, which both stalls that request and makes the phase
    timing look absurd. Long-running servers should call this at boot; the CLI
    and tests can keep loading lazily on first use.
    """
    _model()


def rerank(question: str, hits: list[dict], top_n: int = config.TOP_K) -> list[dict]:
    """Rescore `hits` against `question` with the cross-encoder and keep the best.

    Attaches a `rerank_score` (float; higher = more relevant) to every hit,
    sorts by it descending, and returns the top `top_n`. A no-op on empty input.
    The returned hits keep their original fields (content, source, chunk_index,
    vector_distance, …) so the gate and prompt-builder still work unchanged.
    """
    if not hits:
        return hits

    pairs = [(question, h["content"]) for h in hits]
    scores = _model().predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_n]
