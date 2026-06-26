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
MAX_DISTANCE = 0.40   # cosine DISTANCE floor; <= keep, > discard. Tuned empirically on
                       # this corpus: on-topic queries land <=0.31, off-topic >=0.46, so
                       # 0.40 cleanly separates them (raise if legit answers get dropped,
                       # lower if junk gets through). The grounded prompt is the backstop.
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
