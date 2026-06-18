"""Redis vector index: create, store chunks, and KNN search (via redisvl).

Note: redisvl returns `vector_distance` (cosine distance, lower = closer). Do NOT
treat it as a similarity score.
"""

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
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


def get_index() -> SearchIndex:
    """Return a redisvl SearchIndex connected to REDIS_URL."""
    return SearchIndex.from_dict(SCHEMA, redis_url=config.REDIS_URL)


def create_index(overwrite: bool = True) -> None:
    """Create the index from SCHEMA.

    drop=True also removes any existing chunk keys, so re-ingesting starts clean
    (no stale documents, no duplicates).
    """
    get_index().create(overwrite=overwrite, drop=True)


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
    from redisvl.query import FilterQuery

    idx = get_index()
    query = FilterQuery(
        filter_expression="*",
        return_fields=["content", "source", "chunk_index"],
        num_results=10000,
    )
    results = idx.query(query)
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


def search(
    query_vector: list[float],
    k: int = config.TOP_K,
    source: str | None = None,
) -> list[dict]:
    """KNN search. Return content, source, chunk_index, vector_distance (lower = closer).

    If `source` is given, the search is restricted to chunks from that file
    (metadata filtering: vector similarity combined with a tag filter).
    """
    query = VectorQuery(
        vector=query_vector,
        vector_field_name="embedding",
        num_results=k,
        return_fields=["content", "source", "chunk_index"],
        return_score=True,
        filter_expression=(Tag("source") == source) if source else None,
    )
    results = get_index().query(query)
    return [
        {
            "content": r.get("content"),
            "source": r.get("source"),
            "chunk_index": r.get("chunk_index"),
            "vector_distance": float(r["vector_distance"]),
        }
        for r in results
    ]
