# Python Simple RAG PoC

This is a simple POC for RAG using Python and other tech stack.

## Tech Stack
- Vector Store: Redis Stack (redisvl + redis-py)
- Embeddings: An embedding model (e.g. OpenAI text-embedding-3-small, or a local sentence-transformers model )
- LLM (generation): <to be decided later>
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
