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
