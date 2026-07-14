"""Central configuration for the RAG POC. Every module imports settings from here."""

# Embeddings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking (character-based proxy for tokens)
CHUNK_SIZE = 1800     # ~450 tokens, safely under bge's 512-token cap
CHUNK_OVERLAP = 270   # ~15%

# Redis
REDIS_URL = "redis://localhost:6380"   # 6379 is used by another project
INDEX_NAME = "rag_chunks"
INDEX_PREFIX = "chunk"

# Retrieval
TOP_K = 5
MAX_DISTANCE = 0.40   # cosine DISTANCE floor, applied to the BEST hit only: it answers
                       # "does the corpus contain anything relevant at all?" If even the
                       # closest chunk is beyond this, we short-circuit to "no info".
                       # Off-topic questions bottom out around 0.47 on this corpus, so
                       # 0.40 still refuses them.
RELEVANCE_MARGIN = 0.15  # Once the best hit clears MAX_DISTANCE the query IS on-topic, so
                       # supporting chunks are kept RELATIVE to that best hit (best + margin)
                       # rather than re-judged against the absolute floor. A dense document
                       # (a CV, a spec) spreads its answer across chunks that sit further out
                       # than any single-topic prose chunk; judging them absolutely dropped
                       # the very chunk holding the answer while keeping the useless header.
                       # See store.apply_threshold.
SEARCH_MODE = "hybrid"  # "vector" (pure vector KNN) or "hybrid" (BM25 + vector combined)
HYBRID_ALPHA = 0.7      # weight of vector score in hybrid search (0.0 = text only, 1.0 = vector only)

# Generation
GEN_MODEL = "llama-3.1-8b-instant"   # Groq
TEMPERATURE = 0.0

# Web UI
WEB_HOST = "127.0.0.1"   # localhost only; avoids exposing the Flask debug server to the LAN
WEB_PORT = 5555

# Ingestion
EXCLUDE_PREFIX = "exclude-"
