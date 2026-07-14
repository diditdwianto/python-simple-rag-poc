"""PDF -> Markdown conversion, run once on the way in.

The rest of the pipeline only ever sees text, so rather than teach chunking and
ingestion about PDFs, an uploaded PDF is converted to a Markdown file in data/
and from there it is just another document: chunked, embedded, cited by filename.

Why this file is more careful than "call to_markdown() and write the result":

In RAG, silently losing text is the worst possible failure. A dropped sentence
doesn't raise — it just makes a fact permanently unretrievable while the index
looks perfectly healthy, and the model answers "I don't have enough information"
about something you know you uploaded.

pymupdf4llm has two extraction paths: an ML layout model (`pymupdf_layout`, used
automatically when installed) and the classic font-size heuristic. Measured over
13 real PDFs, *each path drops text on documents the other handles fine* — the
layout model lost 17% of one scanned-ish form, and the heuristic lost 39% of a
menu with an unusual multi-column layout. Neither is safe to trust blindly.

So we run both, count how many of the PDF's words survived each, and keep the
one that actually preserved the document — falling back to raw text extraction
if both mangle it. Structure is nice; completeness is the requirement.
"""

import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
import pymupdf4llm

from src import config

# Fraction of words that may go missing before we stop trusting a conversion.
MAX_WORD_LOSS = 0.02

# Losses within this margin count as a tie, and ties go to the layout model —
# it produces markedly better headings and tables, which chunk better.
TIE_MARGIN = 0.005

# pymupdf4llm.use_layout() flips a module-level global, so conversions must not
# interleave. Flask serves requests on threads, and ingest may convert several
# PDFs in a row.
_layout_lock = threading.Lock()

# The ML layout model lives in a separate package (a dependency of pymupdf4llm
# >= 1.28, but keep working if it's absent).
try:
    import pymupdf.layout  # noqa: F401

    _LAYOUT_AVAILABLE = True
except ImportError:
    _LAYOUT_AVAILABLE = False


def _words(text: str) -> list[str]:
    """Comparable word tokens: ligatures folded (ﬁ -> fi), case and punctuation dropped."""
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def _word_loss(raw: str, md: str) -> float:
    """Fraction of the PDF's words that didn't survive into `md`.

    Conversion is allowed to *rearrange* text; it is not allowed to *lose* it.
    Comparing word multisets catches loss while ignoring reflowing, heading
    markers and table pipes.
    """
    raw_words = Counter(_words(raw))
    if not raw_words:
        return 0.0
    lost = raw_words - Counter(_words(md))
    return sum(lost.values()) / sum(raw_words.values())


def _to_markdown(pdf_path: Path, use_layout: bool) -> str:
    """One extraction pass. Reopens the document: to_markdown() consumes it."""
    with _layout_lock:
        pymupdf4llm.use_layout(use_layout)
        with pymupdf.open(pdf_path) as doc:
            return pymupdf4llm.to_markdown(
                doc,
                # Embeddings are text-only: a picture contributes nothing
                # retrievable but leaves alt-text and file-path noise in chunks.
                ignore_images=True,
                ignore_graphics=True,
                write_images=False,
                show_progress=False,
            ).strip()


def _plain_text(doc: "pymupdf.Document") -> str:
    """Last-resort fallback: raw page text. No structure, but nothing missing."""
    return "\n\n".join(
        f"## Page {i}\n\n{text}"
        for i, page in enumerate(doc, start=1)
        if (text := page.get_text().strip())
    )


def _derive_title(doc: "pymupdf.Document", pdf_path: Path) -> str:
    """The PDF's own metadata title if it set one, else the filename."""
    title = (doc.metadata or {}).get("title", "").strip()
    return title or pdf_path.stem.replace("-", " ").replace("_", " ").title()


def pdf_to_markdown(pdf_path: Path, verbose: bool = True) -> str:
    """Extract `pdf_path` as Markdown, choosing whichever pass preserves the text."""
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError(f"'{pdf_path.name}' is password-protected and can't be read.")
        if doc.page_count == 0:
            raise ValueError(f"'{pdf_path.name}' has no pages.")

        raw = "\n".join(page.get_text() for page in doc)
        if not raw.strip():
            raise ValueError(
                f"No text could be extracted from '{pdf_path.name}'. "
                "It's likely a scanned/image-only PDF, which needs OCR."
            )

        # Score each available extraction path by how much of the document it kept.
        candidates = []
        for use_layout in ([True, False] if _LAYOUT_AVAILABLE else [False]):
            try:
                md = _to_markdown(pdf_path, use_layout)
            except Exception as exc:  # a broken path shouldn't sink the good one
                if verbose:
                    print(f"  [WARN] {pdf_path.name}: layout={use_layout} pass failed: {exc}")
                continue
            if md:
                candidates.append((_word_loss(raw, md), use_layout, md))

        if not candidates:
            best_loss, md = 1.0, ""
        else:
            # Lowest loss wins; near-ties go to the layout model for its better structure.
            best_loss = min(loss for loss, _, _ in candidates)
            winners = [c for c in candidates if c[0] <= best_loss + TIE_MARGIN]
            best_loss, use_layout, md = max(winners, key=lambda c: c[1])
            if verbose:
                scores = ", ".join(
                    f"layout={'on' if lay else 'off'} lost {loss:.1%}"
                    for loss, lay, _ in candidates
                )
                print(
                    f"  [PDF] {pdf_path.name}: {scores} -> using "
                    f"layout={'on' if use_layout else 'off'}"
                )

        if best_loss > MAX_WORD_LOSS:
            if verbose:
                print(
                    f"  [WARN] {pdf_path.name}: every structured pass dropped text "
                    f"(best {best_loss:.1%}) — falling back to plain text extraction."
                )
            md = _plain_text(doc)

        # The knowledge catalog titles each document by its first H1 (see
        # query._title_and_summary). Give it one if the PDF's layout produced none.
        if not md.lstrip().startswith("# "):
            md = f"# {_derive_title(doc, pdf_path)}\n\n{md}"

    return md


def convert_pdf(pdf_path: Path, out_dir: Path | None = None) -> Path:
    """Convert `pdf_path` to a sibling `.md` file and return the Markdown path.

    Written into `out_dir` (default: the PDF's own directory), named after the
    PDF's stem, so `report.pdf` becomes `report.md` and answers cite
    `[source: report.md]`.
    """
    pdf_path = Path(pdf_path)
    target_dir = Path(out_dir) if out_dir else pdf_path.parent
    md_path = target_dir / f"{pdf_path.stem}.md"

    if md_path.name.startswith(config.EXCLUDE_PREFIX):
        raise ValueError(
            f"'{md_path.name}' would be excluded by prefix '{config.EXCLUDE_PREFIX}'."
        )

    md_path.write_text(pdf_to_markdown(pdf_path), encoding="utf-8")
    return md_path
