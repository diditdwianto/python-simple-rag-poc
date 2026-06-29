"""Ingest pipeline: load data/ documents -> chunk -> embed -> store in Redis.

Run:  python -m src.ingest
"""

from pathlib import Path

from dotenv import load_dotenv

from src.chunking import chunk_text
from src.embeddings import embed_documents
from src.store import add_chunks, create_index, delete_source, ensure_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_SUFFIXES = {".txt", ".md"}


def _store_file(path: Path) -> int:
    """Chunk, embed and store one file's chunks. Returns the count stored.

    Assumes any prior chunks for this filename have already been cleared (a fresh
    full ingest drops the whole index; incremental ingest calls delete_source).
    """
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    vectors = embed_documents(chunks)

    records = [
        {
            "content": chunk,
            "source": path.name,
            "chunk_index": i,
            "embedding": vec,  # list[float]; store.add_chunks converts to bytes
        }
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    add_chunks(records)
    return len(records)


def ingest_file(filename: str) -> dict:
    """Incrementally ingest a single file from data/ into the existing index.

    Unlike main(), this does NOT drop the index — it adds (or replaces, if the
    same filename was ingested before) just this one file. Used by the web UI
    after an upload. On the next full `python -m src.ingest`, the file is picked
    up like any other in data/.
    """
    from src import config

    load_dotenv()
    path = DATA_DIR / filename

    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Allowed: .md, .txt")
    if not path.is_file():
        raise FileNotFoundError(f"No such file in data/: {filename}")
    if path.name.startswith(config.EXCLUDE_PREFIX):
        raise ValueError(
            f"'{filename}' is excluded by prefix '{config.EXCLUDE_PREFIX}' and won't be ingested."
        )

    ensure_index()
    replaced = delete_source(path.name)
    count = _store_file(path)
    return {"source": path.name, "chunks": count, "replaced": replaced}


def main() -> None:
    from src import config

    load_dotenv()
    create_index(overwrite=True)

    skipped = []
    total = 0
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name.startswith(config.EXCLUDE_PREFIX):
            skipped.append(path.name)
            print(f"  [SKIP] {path.name} (excluded by prefix '{config.EXCLUDE_PREFIX}')")
            continue

        count = _store_file(path)
        total += count
        print(f"  [INGEST] {path.name}: {count} chunks")

    print(f"\nTotal chunks stored: {total}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
