"""Ingest pipeline: load data/ documents -> chunk -> embed -> store in Redis.

Run:  python -m src.ingest
"""

from pathlib import Path

from dotenv import load_dotenv

from src.chunking import chunk_text
from src.embeddings import embed_documents
from src.pdf import convert_pdf
from src.store import add_chunks, create_index, delete_source, ensure_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
# What an upload is allowed to be. PDFs never reach the chunker directly — they
# are converted to Markdown first (see ensure_markdown), and the .md is ingested.
UPLOAD_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def ensure_markdown(pdf_path: Path) -> tuple[Path, bool]:
    """Return (markdown path, converted?) for `pdf_path`, converting if the .md is
    missing or older than the PDF.

    The .md sits next to the PDF in data/ and is the file that actually gets
    indexed, so it's also the filename the LLM cites and the one the user can
    inspect on the /data page.
    """
    md_path = pdf_path.with_suffix(".md")
    if md_path.is_file() and md_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return md_path, False
    return convert_pdf(pdf_path), True


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

    if path.suffix.lower() not in UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Allowed: .md, .txt, .pdf")
    if not path.is_file():
        raise FileNotFoundError(f"No such file in data/: {filename}")
    if path.name.startswith(config.EXCLUDE_PREFIX):
        raise ValueError(
            f"'{filename}' is excluded by prefix '{config.EXCLUDE_PREFIX}' and won't be ingested."
        )

    if path.suffix.lower() in PDF_SUFFIXES:
        path, _ = ensure_markdown(path)

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

    # PDFs are converted up front so the loop below sees only Markdown. A PDF
    # dropped into data/ by hand therefore behaves exactly like one uploaded
    # through the web UI.
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        if pdf.name.startswith(config.EXCLUDE_PREFIX):
            continue
        md, converted = ensure_markdown(pdf)
        if converted:
            print(f"  [CONVERT] {pdf.name} -> {md.name}")

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
