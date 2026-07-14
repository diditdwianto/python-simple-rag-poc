"""Redis vector index: create, store chunks, and search (via redisvl).

Supports two search modes:
- vector: pure KNN vector similarity search
- hybrid: BM25 text search + vector similarity, combined with weighted scoring

Note: redisvl returns `vector_distance` (cosine distance, lower = closer). Do NOT
treat it as a similarity score.
"""

import re
from functools import lru_cache

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

from src import config

SCHEMA = {
    "index": {
        "name": config.INDEX_NAME,
        "prefix": config.INDEX_PREFIX,
        "storage_type": "hash",
    },
    "fields": [
        {"name": "content", "type": "text"},
        {"name": "source", "type": "tag"},
        {"name": "chunk_index", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": config.EMBED_DIM,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
}

# Redis full-text special characters. Stripped from free-form queries and
# backslash-escaped inside tag filters, so a raw question (or a filename with
# punctuation like "alien-snacks.md") can never produce an invalid FT.SEARCH.
_FT_SPECIAL = re.compile(r"""[,.<>{}\[\]"':;!@#$%^&*()\-+=~|/\\]""")


def _sanitize_query(text: str) -> str:
    """Turn a free-form question into a safe full-text query (special chars -> space)."""
    return " ".join(_FT_SPECIAL.sub(" ", text).split())


def _escape_tag(value: str) -> str:
    """Backslash-escape a tag value (e.g. a filename) for an FT tag filter."""
    return _FT_SPECIAL.sub(lambda m: "\\" + m.group(0), value)


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    """Return a redisvl SearchIndex connected to REDIS_URL (cached)."""
    return SearchIndex.from_dict(SCHEMA, redis_url=config.REDIS_URL)


@lru_cache(maxsize=1)
def _raw_client():
    """Cached low-level redis-py client for the hand-written BM25 FT.SEARCH query."""
    from redis import Redis

    return Redis.from_url(config.REDIS_URL, decode_responses=False)


def ping() -> None:
    """Touch the index connection (first call establishes it). Used to time the
    'loading index' phase of a query."""
    get_index()
    _raw_client().ping()


def create_index(overwrite: bool = True) -> None:
    """Create the index from SCHEMA.

    drop=True also removes any existing chunk keys, so re-ingesting starts clean
    (no stale documents, no duplicates).
    """
    get_index().create(overwrite=overwrite, drop=True)


def ensure_index() -> None:
    """Create the index only if it does not already exist. Never drops data.

    Used by incremental (single-file) ingestion, which must add to the existing
    index rather than wiping it like create_index(overwrite=True) does.
    """
    index = get_index()
    if not index.exists():
        index.create(overwrite=False, drop=False)


def delete_source(source: str) -> int:
    """Remove every chunk belonging to one source filename. Returns how many were
    deleted. Lets a single file be re-ingested without duplicating its chunks."""
    query = FilterQuery(
        filter_expression=(Tag("source") == source),
        return_fields=["source"],
        num_results=10000,
    )
    keys = [r["id"] for r in get_index().query(query) if r.get("id")]
    if keys:
        get_index().drop_keys(keys)
    return len(keys)


def add_chunks(records: list[dict]) -> None:
    """Store chunk records.

    Each record: {content, source, chunk_index, embedding(list[float])}.
    This is the single place the float32 byte conversion happens.
    """
    payload = [
        {
            "content": r["content"],
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "embedding": array_to_buffer(r["embedding"], dtype="float32"),
        }
        for r in records
    ]
    get_index().load(payload)


def fetch_all() -> list[dict]:
    """Return all chunks from the index, sorted by source then chunk_index."""
    query = FilterQuery(
        filter_expression="*",
        return_fields=["content", "source", "chunk_index"],
        num_results=10000,
    )
    results = get_index().query(query)
    chunks = [
        {
            "content": r.get("content"),
            "source": r.get("source"),
            "chunk_index": int(r.get("chunk_index", 0)),
        }
        for r in results
    ]
    chunks.sort(key=lambda c: (c["source"], c["chunk_index"]))
    return chunks


def apply_threshold(hits: list[dict]) -> list[dict]:
    """Relevance gate shared by every query path (the 'no info' short-circuit).

    This makes TWO decisions, and they deliberately use different rules:

    1. *Is the query on-topic at all?* Judged on the BEST hit against the
       absolute floor MAX_DISTANCE. If even the closest chunk is beyond it,
       nothing is returned and the caller answers "I don't have enough
       information". This is what keeps junk questions out.

    2. *Which chunks support the answer?* Judged RELATIVE to that best hit
       (best + RELEVANCE_MARGIN). Once decision 1 says the corpus is relevant,
       re-applying the absolute floor here is wrong: in a dense document the
       answer often lives in a chunk that ranks 2nd and sits well past the
       floor, so the absolute test discarded the chunk containing the answer
       and kept only the document's header. Rank was right; the cutoff wasn't.

    A hit needs vector support to survive. Hits found by BM25 alone (no
    `vector_distance`) used to ride along, but hybrid_search normalises text
    scores *within* each result set, so the top BM25 hit always scores 1.0 even
    when it matched nothing but filler words — that let a chunk with zero
    semantic relevance outrank the chunk holding the actual answer. BM25 still
    earns its keep by *ranking* vector-supported hits (combined_score); it just
    no longer admits chunks on its own.
    """
    distances = [h["vector_distance"] for h in hits if h.get("vector_distance") is not None]
    if not distances or min(distances) > config.MAX_DISTANCE:
        return []

    cutoff = min(distances) + config.RELEVANCE_MARGIN
    return [
        h
        for h in hits
        if h.get("vector_distance") is not None and h["vector_distance"] <= cutoff
    ]


def search(
    query_vector: list[float],
    k: int = config.TOP_K,
    source: str | None = None,
    query_text: str | None = None,
) -> list[dict]:
    """Search chunks. Dispatches to vector or hybrid based on config.SEARCH_MODE.

    If query_text is provided and SEARCH_MODE is "hybrid", uses BM25+vector.
    Otherwise falls back to pure vector KNN search.
    """
    if config.SEARCH_MODE == "hybrid" and query_text:
        return hybrid_search(query_text, query_vector, k=k, source=source)
    return vector_search(query_vector, k=k, source=source)


def _normalize(scores: dict) -> dict:
    """Scale a {id: score} map to 0..1. Equal scores all map to 1.0 (equally relevant)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {key: 1.0 for key in scores}
    return {key: (v - lo) / (hi - lo) for key, v in scores.items()}


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    k: int = config.TOP_K,
    source: str | None = None,
    alpha: float = config.HYBRID_ALPHA,
) -> list[dict]:
    """Hybrid BM25 text + vector search, combined in Python.

    Runs a separate BM25 text query and vector KNN query, normalizes scores,
    then merges with weighted combination: final = alpha * vec_score + (1-alpha) * text_score.

    Alpha controls the blend: 0.0 = text only, 1.0 = vector only, 0.7 = mostly vector
    with a text boost. Each result keeps the raw `vector_distance` of its vector
    match (None if found by text only) so apply_threshold() can still gate on it.
    """
    fetch_k = k * 3

    text_hits = _text_search(query_text, source=source, k=fetch_k)
    vec_hits = vector_search(query_vector, k=fetch_k, source=source)

    text_scores = {h["id"]: h["text_score"] for h in text_hits}
    vec_scores = {h["id"]: h["vec_score"] for h in vec_hits}
    vec_dist = {h["id"]: h["vector_distance"] for h in vec_hits}
    all_ids = set(text_scores) | set(vec_scores)

    content_map = {}
    for h in (*text_hits, *vec_hits):
        content_map[h["id"]] = {
            "content": h["content"],
            "source": h["source"],
            "chunk_index": h["chunk_index"],
        }

    norm_text = _normalize(text_scores)
    norm_vec = _normalize(vec_scores)

    combined = []
    for doc_id in all_ids:
        ts = norm_text.get(doc_id, 0.0)
        vs = norm_vec.get(doc_id, 0.0)
        score = alpha * vs + (1 - alpha) * ts
        meta = content_map[doc_id]
        combined.append({
            "content": meta["content"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "combined_score": round(score, 6),
            "vector_distance": vec_dist.get(doc_id),  # None if text-only match
            "id": doc_id,
        })

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    return combined[:k]


def _text_search(query_text: str, source: str | None = None, k: int = 20) -> list[dict]:
    """BM25 full-text search on the content field.

    Terms are OR-ed. RediSearch treats space-separated terms inside a field as an
    intersection, which for a natural-language question means "every word must
    appear in the chunk" — no chunk ever contains 'how', 'many' AND 'stickearn',
    so BM25 silently returned ZERO hits for every question and `SEARCH_MODE =
    "hybrid"` quietly degraded to pure vector search. OR-ing restores recall and
    lets BM25's IDF weighting do the ranking: a rare term like a company name
    scores far above filler words like "how".
    """
    from redis.commands.search.query import Query as FTQuery

    terms = _sanitize_query(query_text).split()
    if not terms:
        return []  # nothing searchable left (e.g. query was all punctuation)
    cleaned = "|".join(terms)

    if source:
        q_str = f"(@source:{{{_escape_tag(source)}}}) (@content:({cleaned}))"
    else:
        q_str = f"@content:({cleaned})"

    ft_query = (
        FTQuery(q_str)
        .scorer("BM25")
        .return_fields("content", "source", "chunk_index")
        .with_scores()
        .paging(0, k)
        .dialect(2)
    )
    results = _raw_client().ft(config.INDEX_NAME).search(ft_query)

    def _decode(v):
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return v

    hits = []
    for doc in results.docs:
        doc_id = _decode(doc.id)
        text_score = float(doc.score) if hasattr(doc, "score") else 0.0
        src = _decode(doc.source) if hasattr(doc, "source") else ""
        if src.endswith("\x00"):
            src = src[:-1]

        hits.append({
            "id": doc_id,
            "content": _decode(doc.content) if hasattr(doc, "content") else "",
            "source": src,
            "chunk_index": int(_decode(doc.chunk_index)) if hasattr(doc, "chunk_index") else 0,
            "text_score": text_score,
        })
    return hits


def vector_search(
    query_vector: list[float],
    k: int = config.TOP_K,
    source: str | None = None,
) -> list[dict]:
    """Pure KNN vector search. Return content, source, chunk_index, vector_distance, vec_score."""
    query = VectorQuery(
        vector=query_vector,
        vector_field_name="embedding",
        num_results=k,
        return_fields=["content", "source", "chunk_index"],
        return_score=True,
        filter_expression=(Tag("source") == source) if source else None,
    )
    results = get_index().query(query)
    hits = []
    for r in results:
        dist = float(r["vector_distance"])
        # Convert cosine distance to similarity score (1 - distance), higher = better
        vec_score = max(0.0, 1.0 - dist)
        hits.append({
            "content": r.get("content"),
            "source": r.get("source"),
            "chunk_index": r.get("chunk_index"),
            "vector_distance": dist,
            "id": r.get("id", ""),
            "vec_score": vec_score,
        })
    return hits
