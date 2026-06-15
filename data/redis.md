# Redis

Redis is an open-source, in-memory data store created by Salvatore Sanfilippo and
first released in 2009. It is commonly used as a database, cache, and message
broker. Because it keeps data in memory, Redis offers very low latency for reads
and writes.

Redis Stack extends Redis with additional capabilities, including vector search,
full-text search, JSON support, and time-series data. The vector search feature
lets applications store embeddings and run k-nearest-neighbor queries, which makes
Redis suitable as a vector database for retrieval-augmented generation systems.

Redis supports two vector index types: FLAT, which performs an exact brute-force
search, and HNSW, which is an approximate but much faster graph-based index. For
small datasets, FLAT is simple and returns exact results. For very large datasets,
HNSW scales far better.
