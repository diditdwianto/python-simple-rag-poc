# Python Simple RAG PoC

This is a simple POC for RAG using Python and other tech stack.

## Quickstart (how to run)

```bash
# 1. virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. dependencies
pip install -r requirements.txt

# 3. Groq API key (free, from https://console.groq.com)
cp .env.example .env       # then edit .env and set GROQ_API_KEY=...

# 4. start Redis Stack (host port 6380; 6379 is often used by other projects)
docker compose up -d

# 5. ingest the docs in data/
python -m src.ingest

# 6. ask a question
python -m src.query "What vector index types does Redis support?"
```

An out-of-corpus question returns: `I don't have enough information to answer that.`
Put your own `.txt` / `.md` files in `data/` and re-run `python -m src.ingest`.

See `CLAUDE.md` for the architecture and `TASKS.md` for the build breakdown.

## Tech Stack
- Vector Store: Redis Stack (redisvl + redis-py) via Docker
- Embeddings: Local sentence-transformer
    - BAAI/bge-small-en-v1.5 - 328 dimentions, strong retrieval quality, ~130 MB
    - all-MiniLM-L6-v2: 384 dimention, classic lightweight baseline, very fast
- LLM (generation): `llama-3.1-8b-instant` via Groq (free tier) — pluggable backend
- Orchestration: Plain python first; then add LangChain/Llamaindex only if we want abstraction

## The minimal flow in one picture

  INDEX:  docs ─► chunk ─► embed ─► Redis (HNSW index: vector + text + metadata)

  QUERY:  question ─► embed ─► Redis KNN (top-k) ─► prompt(context + question)
                                                         │
                                                         ▼
                                                  LLM (generation) ─► grounded answer

## How the architecture works

### Phase 1 — Indexing (offline, done once / on update):
  
  Documents → Chunk into pieces → Embed each chunk → Store vectors + text in a DB

  1. Load your source docs (PDFs, markdown, DB rows, etc.).
  2. Chunk them into small pieces (e.g. 300–800 tokens). Chunking matters a lot — too big and retrieval is noisy, too
  small and you lose context.
  3. Embed each chunk: an embedding model turns text into a vector (a list of floats) that captures meaning.
  4. Store each vector alongside its original text and metadata in a vector store.

### Phase 2 — Retrieval + Generation (online, per user question):

  Question → Embed → Vector search (top-k similar chunks) → Build prompt → LLM → Answer

  1. Embed the question with the same embedding model.
  2. Search the vector store for the k most similar chunks (cosine / dot-product similarity).
  3. Assemble a prompt: system instructions + the retrieved chunks as context + the user's question.
  4. Generate the answer with an LLM, instructing it to answer only from the provided context and cite sources.

## Why REDIS?

  Redis Stack (or Redis 8+) ships the vector search capability via the search module.
  
  Concretely Redis gives us:

  - Vector storage + ANN search — store embeddings in hashes or JSON, create an index with HNSW or FLAT, and run
  K-nearest-neighbor queries. This is the core of the retrieval step.
  - Metadata filtering — combine vector similarity with tag/numeric filters (e.g. "only docs from 2024").
  - Hybrid search — full-text (BM25) + vector in one query.
  - Caching — Redis's original strength. Two useful layers: a semantic cache (return a stored answer when a new
  question is semantically near a previous one) and an embedding cache (avoid re-embedding identical text).
  - Speed — in-memory, very low latency.

  The official redis Python client plus redisvl (Redis Vector Library) make this ergonomic.

  Trade-off to know: Redis holds vectors in RAM, so for very large corpora (tens of millions of chunks) memory cost
  is the main consideration. For a POC or small/medium production set, it's excellent.

## Technical Details

### Chunking Strategies

Chosen:
- Recursive splitting: Splits on a hierarchy of separators. Try to break on paragraph first; if a piece is still too big, then break on sentences; then on words. Keep natural units together while respecting our size cap.
- Chunk size: ~400 tokens per chunk, ~60 tokens overlap (~15%)

Other strategies not used: 
- Fixed size (every N character / tokens): Cut mid sentence / mid word, crude
- Sentence-based: Sentence may vary wildly in length, chunk size get uneven.
- Structure-aware: split on document structure. Worth if want to ingest markdown.
- Semantic chunking: use embeddings to detect topic shifts and cut there. Highest quality but overkill for a PoC

Metadata to store with each chunk:
- source: filename / URL (so we can cite it in the answer)
- chunk_index: position within the document (lets us fetch neighbors later)
- title / section: if available (improves filtering and citations)

Notes
- Retrieval can only even return chunks we created. If a question's answer is split across two badly-cut chunks, no amount of LLM quality will fix it (the model never sees the whole answer)

How a chunk is stored
- Each chunk become one Redis key holding several fields (because Redis isn't natively a 'vector DB')
    - embedding   = <383 float32 bytes>
    - content     = "The revenue grew 20% in Q4 ..."
    - source      = "annual_report_2025.pdf"
    - chunk_index = 42
- Format: HASH (flat fields, most memory-efficient)
- The index: FT.CREATE
    - We define an index once that tells Redis which field exists and how to search them
    - The vector field is the important one.
    - Conceptually (this is only an example for understanding, not as deliverable):
        - FT.CREATE chunk_idx
            ON HASH PREFIX 1 chunk:
            SCHEMA
              embedding    VECTOR FLAT 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE
              content      TEXT
              source       TAG
              chunk_index  NUMERIC
        - Why FLAT?
            - For PoC purpose, thousands of chunks on Macbook laptop would be an instant, gives exact result, and zero tuning.
            - Accuracy: Exact (always the true top-k)
            - Memory usage: lower than HNSW
            - Speed: Slower as data grows
        - Why COSINE as DISTANCE_METRIC (how 'similarity' is measured between two vectors')?
            - COSINE: angle between vectors (ignores magnitude). Our bge-small produces small embeddings, and cosine is the metric these models are trained and benchmarked on. This is the safe, standard match.
            - IP: inner product / dot product
            - L2: Euclidean distabce 
        - DIM 384: must match bge-small output, otherwise index cration or insert fail
