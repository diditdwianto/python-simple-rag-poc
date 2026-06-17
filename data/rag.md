# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, or RAG, is a technique that improves the answers
of a large language model by retrieving relevant documents at query time and
providing them to the model as context. Instead of relying only on what the model
learned during training, the model answers using the retrieved text. This reduces
hallucination and lets the system use up-to-date, domain-specific information.

A RAG pipeline has two phases. In the indexing phase, documents are split into
chunks, each chunk is converted into a vector by an embedding model, and the
vectors are stored in a vector database. In the query phase, the user's question
is embedded with the same model, the most similar chunks are retrieved, and those
chunks are placed into the prompt alongside the question for the language model to
answer.

Chunking is one of the most important factors in RAG quality. Chunks that are too
large produce blurry, unfocused embeddings, while chunks that are too small lose
surrounding context. A common starting point is a few hundred tokens per chunk
with a small overlap between consecutive chunks.

---

## Ingestion pipeline — how documents are indexed

The ingestion pipeline is run by `python -m src.ingest` (or via the POST /api/ingest
endpoint in the web UI). It performs these steps in order:

1. **Load documents.** Every .txt and .md file in the data/ directory is read as
   UTF-8 text. Each file's filename becomes the source label for all its chunks.

2. **Recreate the Redis index.** `store.create_index(overwrite=True)` drops any
   existing index and creates a fresh one from the schema (HASH storage, FLAT
   algorithm, COSINE distance, DIM 384, FLOAT32 vectors). This ensures no stale
   or duplicate chunks remain.

3. **Chunk each document.** The text is split using
   RecursiveCharacterTextSplitter (chunk_size=1800 chars ≈ 450 tokens,
   chunk_overlap=270 chars ≈ 15%). The splitter respects paragraph → sentence →
   word boundaries. Each chunk gets metadata: source (filename) and chunk_index
   (position within that file).

4. **Embed the chunks.** All chunks from one file are batch-encoded using
   bge-small-en-v1.5 via `embed_documents()` (batch_size=32). No query prefix
   is added — documents are embedded raw. Normalization is always enabled
   (normalize_embeddings=True) because cosine similarity requires unit vectors.

5. **Store in Redis.** Each chunk record contains: content (the plain text),
   source (filename tag), chunk_index (numeric), and embedding (384-dim float32
   vector, converted to bytes by redisvl's array_to_buffer). The records are
   loaded into the index via `SearchIndex.load()`.

The total number of chunks stored is printed at the end. Re-running ingestion
always starts from a clean index (overwrite=True), so there are no duplicates.
To add or update documents, simply put new .txt/.md files in data/ and re-run
ingestion.

## Query pipeline — how a question is answered

When a user submits a question (via CLI, web UI, or API), these steps execute:

1. **Embed the question.** `embed_query()` prepends the bge query prefix
   "Represent this sentence for searching relevant passages:" to the question,
   then encodes it with the same bge-small-en-v1.5 model (normalized).

2. **Retrieve from Redis.** `store.search()` runs a KNN query (top_k=5 by
   default) against the FLAT index. If a source filter is provided, a tag
   filter narrows results to that file only.

3. **Distance threshold.** Hits with vector_distance > MAX_DISTANCE (0.6) are
   discarded. If no hits remain, the system returns "I don't have enough
   information to answer that." without calling the LLM at all.

4. **Build the grounded prompt.** Surviving hits are formatted as labeled
   context blocks: [source: filename.md] followed by the chunk text. A system
   message instructs the model to answer only from the context, cite sources,
   and say "I don't have enough information" if the answer isn't there.

5. **Generate the answer.** The prompt is sent to llama-3.1-8b-instant on Groq
   (temperature=0). The model returns a concise, source-cited answer grounded
   in the retrieved context.

## This project's tech stack

| Layer | Choice | Cost | Notes |
|---|---|---|---|
| Web UI | Flask | Free | Dark-themed chat interface at localhost:5555. Provides /api/query, /api/ingest, /api/status endpoints. |
| Embeddings | BAAI/bge-small-en-v1.5 (local, sentence-transformers) | Free | 384 dimensions. Must normalize embeddings. Query/document asymmetry: documents get no prefix, queries prepend "Represent this sentence for searching relevant passages:". |
| Vector store | Redis Stack (Docker, local) via redisvl | Free | HASH storage, FLAT algorithm, COSINE distance, FLOAT32 vectors, DIM 384. Fields: content (TEXT), source (TAG), chunk_index (NUMERIC), embedding (VECTOR). |
| Generation | llama-3.1-8b-instant via Groq (free tier) | Free | Pluggable backend — swapping to Ollama + gemma3:4b is a one-file change (generate.py). Temperature 0 for factual answers. |
| Chunking | RecursiveCharacterTextSplitter (langchain) | Free | Chunk size 1800 chars (~450 tokens), overlap 270 chars (~15%). Respects paragraph → sentence → word hierarchy. |
| Orchestration | Plain Python | Free | No LangChain/LlamaIndex orchestration layer. |

## Embedding model details — bge-small-en-v1.5

The embedding model is BAAI/bge-small-en-v1.5, loaded locally via sentence-transformers.
It produces 384-dimensional vectors. Normalization is always enabled (normalize_embeddings=True)
because bge is trained for cosine similarity which depends on normalized vectors.

There is a critical query/document asymmetry: when embedding documents for storage, the raw
chunk text is used with no prefix. When embedding a search query, the prefix string
"Represent this sentence for searching relevant passages:" must be prepended to the question
before encoding. The same model is used for both index and query time. Mixing this up is the
most common bug in bge-based RAG systems.

At index time, documents are batch-encoded (32-64 at a time) for throughput. At query time,
a single question is encoded, which takes only a few milliseconds.

## Redis vector store configuration

The Redis vector index uses the following schema:

- Storage type: HASH (flat fields, memory-efficient)
- Algorithm: FLAT (exact top-k results, zero tuning needed at POC scale; switch to HNSW
  with M=16, EF_CONSTRUCTION=200, EF_RUNTIME=10 only when corpus reaches hundreds of thousands)
- Distance metric: COSINE (matches normalized bge embeddings)
- Vector type: FLOAT32, dimension 384 (must match the embedder output exactly)
- Fields: content (TEXT for optional hybrid search), source (TAG for exact-match filtering
  and citations), chunk_index (NUMERIC for range filters), embedding (VECTOR)

The redisvl library (Redis Vector Library) is used instead of hand-writing FT.CREATE /
FT.SEARCH commands — it handles float32 byte-packing and query building. Memory usage is
approximately 3 KB per chunk, so 10k chunks consume roughly 30 MB of RAM.

KNN queries use the shape: *=>[KNN 5 @embedding $vec AS score], sorted by score.
For filtered queries (restricting to one source file):
(@source:{file.pdf})=>[KNN 5 @embedding $vec AS score].

The default retrieval parameters are TOP_K=5 and MAX_DISTANCE=0.6 (cosine distance,
lower is closer). The distance threshold is a coarse pre-filter gate; the grounded
system prompt is the real guard against off-topic answers.

## Chunking strategy

Recursive splitting is used via langchain's RecursiveCharacterTextSplitter. It tries
to break on paragraphs first; if a piece is still too large, it breaks on sentences,
then on words. This keeps natural units together while respecting the size cap.

- Chunk size: 1800 characters (approximately 450 tokens, safely under bge-small's
  512-token limit)
- Overlap: 270 characters (approximately 15%) so boundary-straddling context survives
- Metadata per chunk: source (filename), chunk_index (position in document)

Other chunking strategies considered but not used:
- Fixed-size splitting (every N characters/tokens): cuts mid-sentence, crude
- Sentence-based: sentence lengths vary wildly, producing uneven chunk sizes
- Structure-aware: splits on document headings — useful for markdown-heavy docs
- Semantic chunking: uses embeddings to detect topic shifts — highest quality but
  overkill for a POC

Retrieval can only return chunks that were created during indexing. If an answer is
split across two badly-cut chunks, no LLM can recover it. Tune chunk size first if
answers feel vague (too big) or fragmented (too small).

## LLM generation — grounded prompt design

The generation step uses a grounded prompting approach. The system prompt instructs
the model to answer using only the provided context. If the answer is not in the
context, the model must reply exactly: "I don't have enough information to answer that."
The model is told not to guess and to cite the [source] label(s) it used.

Context is formatted with source labels so the model can cite them:
[source: report_2024.pdf]
<chunk text>

Three safeguards are in place:
1. Score threshold — if the top KNN hit exceeds the MAX_DISTANCE threshold, the system
   short-circuits to "no info" instead of feeding junk context to the LLM.
2. Most-relevant-first ordering — KNN results are already sorted by similarity score.
3. Context is treated as untrusted data, not instructions — rules stay in the system
   message and context is clearly labeled, which mitigates prompt injection.

The generation model is llama-3.1-8b-instant on Groq with temperature 0. The system
prompt is passed as a separate system chat message. The generation backend is pluggable:
swapping to Ollama + gemma3:4b or any other model requires changing only generate.py
and the GEN_MODEL config value.

## Web UI

The project includes a Flask-based web interface for querying the RAG system from
a browser. It runs on port 5555 (configurable in config.py as WEB_PORT).

API endpoints:
- GET / — serves the chat UI
- POST /api/query — accepts JSON {question, source?}, returns {answer, sources, context}
- POST /api/ingest — triggers document re-ingestion
- GET /api/status — returns index name, document count, and record count

The UI features a dark theme with a chat-style interface. Users type questions, see
answers with source citations, and can expand retrieved context chunks to inspect
distance scores and chunk content. A source filter input allows restricting retrieval
to a specific document file. A "Re-ingest Docs" button triggers the full indexing
pipeline from the browser.

## Key invariants

- The embedding model dimension (384) must exactly match the Redis index DIM.
- The same embedding model must be used for both indexing and querying.
- Query prefix is prepended to questions only; documents get no prefix. Both are normalized.
- Always check the retrieval score (MAX_DISTANCE threshold) before generating an answer.
- The generation backend is swappable via generate.py — no other module should import
  Groq directly.
