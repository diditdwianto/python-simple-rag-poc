"""Local embeddings via bge-small.

Document and query embeddings are asymmetric for bge: documents get NO prefix,
queries get config.QUERY_PREFIX. Both are normalized. Same model for both.
"""

from sentence_transformers import SentenceTransformer

from src import config

_model = SentenceTransformer(config.EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage. NO prefix. Normalized. Batched."""
    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question. Prepend config.QUERY_PREFIX. Normalized."""
    vector = _model.encode(
        config.QUERY_PREFIX + text,
        normalize_embeddings=True,
    )
    return vector.tolist()
