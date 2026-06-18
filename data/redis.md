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

## Redis Search concepts explained with RDBMS analogies

If you are familiar with relational databases (like PostgreSQL or MySQL), here is
how Redis Search concepts map to RDBMS terms:

| Redis Search concept | RDBMS analogy |
|---|---|
| Index | Table (with full-text + vector columns) |
| Document (a HASH key like chunk:1) | Row |
| num_docs (e.g. 17) | Number of rows |
| num_records (e.g. 3086) | Total word-to-row mappings in the inverted index. Think of it like how many rows you would have in a junction table if you broke every word from every row's text content into its own row pointing back to the parent row. |
| Field (content, source, chunk_index, embedding) | Column |
| TAG field (source) | Indexed VARCHAR with exact-match lookups (like WHERE source = 'file.md') |
| TEXT field (content) | Full-text indexed TEXT column (like a GIN index in PostgreSQL) |
| VECTOR field (embedding) | A special column that stores a 384-float array and supports nearest-neighbor queries |
| KNN search | SELECT * FROM chunks ORDER BY cosine_distance(embedding, ?) LIMIT 5 |
| Tag filter (@source:{file.md}) | WHERE source = 'file.md' |
| Hybrid search | Full-text search combined with vector search in one query |

In this project, the Redis index stores 17 chunks (rows). Each chunk has a text
column for its content, a tag column for its source filename, a numeric column
for its chunk position, and a 384-dimensional vector column for its embedding.
The 3086 indexed terms represent all the unique word-to-row postings across
every chunk's text content combined — similar to the internal structure of a
full-text search index in a relational database.
