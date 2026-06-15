"""Ingest pipeline: load data/ documents -> chunk -> embed -> store in Redis.

Run:  python -m src.ingest
"""

from pathlib import Path

from dotenv import load_dotenv

from src.chunking import chunk_text
from src.embeddings import embed_documents
from src.store import add_chunks, create_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_SUFFIXES = {".txt", ".md"}


def main() -> None:
    load_dotenv()
    create_index(overwrite=True)

    total = 0
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue

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
        total += len(records)
        print(f"  {path.name}: {len(records)} chunks")

    print(f"Total chunks stored: {total}")


if __name__ == "__main__":
    main()
