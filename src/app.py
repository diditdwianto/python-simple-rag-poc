"""Flask web UI for the RAG POC.

Run:  python -m src.app
"""

import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src import config
from src.embeddings import embed_query
from src.generate import generate_stream, SYSTEM_PROMPT
from src.ingest import (
    DATA_DIR,
    PDF_SUFFIXES,
    UPLOAD_SUFFIXES,
    ensure_markdown,
    ingest_file,
)
from src.query import (
    NO_INFO,
    build_catalog,
    build_user_prompt,
    format_catalog_answer,
    is_catalog_question,
)
from src.rerank import rerank, warmup as warmup_reranker
from src.store import apply_threshold, fetch_all, ping, search

load_dotenv()

app = Flask(__name__)

# Long-running server: load the reranker at boot so the first query's "Reranking"
# phase reflects actual reranking, not the one-time model load. (bge-small already
# loads eagerly at import; this keeps the two consistent. CLI/tests load lazily.)
if config.RERANK_ENABLED:
    warmup_reranker()


@app.route("/")
def index():
    return render_template("index.html")


def _context_data(hits: list[dict]) -> list[dict]:
    """Shape retrieved hits for the UI's 'retrieved context' panel."""
    return [
        {
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            "content": h["content"],
            "distance": round(
                h["combined_score"] if "combined_score" in h else h["vector_distance"], 4
            ),
            "search_mode": "hybrid" if "combined_score" in h else "vector",
            # Present only when stage-2 reranking ran; the UI shows it when set.
            "rerank_score": round(h["rerank_score"], 3) if "rerank_score" in h else None,
        }
        for h in hits
    ]


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    source = data.get("source") or None

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        overall = time.perf_counter()
        steps = []

        # The pipeline is timed phase-by-phase so the answer can carry a detailed
        # activity breakdown (shown collapsed at the bottom of the answer).
        t = time.perf_counter()
        ping()
        steps.append({"label": "Loading index", "ms": (time.perf_counter() - t) * 1000})

        # "What do you know?" style questions are answered from a catalog computed
        # live from the index — no retrieval, no LLM, never out of date.
        if is_catalog_question(question):
            t = time.perf_counter()
            catalog = build_catalog()
            steps.append({
                "label": "Building knowledge catalog",
                "ms": (time.perf_counter() - t) * 1000,
                "detail": f"{len(catalog)} documents",
            })
            return jsonify({
                "answer": format_catalog_answer(catalog),
                "sources": [c["source"] for c in catalog],
                "context": [],
                "activity": {"steps": steps, "total_ms": (time.perf_counter() - overall) * 1000},
            })

        t = time.perf_counter()
        qvec = embed_query(question)
        steps.append({"label": "Embedding query", "ms": (time.perf_counter() - t) * 1000})

        # Stage 1 (recall): pull a wide candidate set when reranking will narrow it.
        # Mirrors query.retrieve() — kept inline here to time each phase separately.
        t = time.perf_counter()
        fetch_k = config.RERANK_FETCH_K if config.RERANK_ENABLED else config.TOP_K
        candidates = search(qvec, k=fetch_k, source=source, query_text=question)
        steps.append({
            "label": f"Searching index ({config.SEARCH_MODE})",
            "ms": (time.perf_counter() - t) * 1000,
            "detail": f"{len(candidates)} candidates",
        })

        # Stage 2 (precision): cross-encoder rescores the candidates, keeps TOP_K.
        if config.RERANK_ENABLED:
            t = time.perf_counter()
            candidates = rerank(question, candidates, top_n=config.TOP_K)
            steps.append({
                "label": f"Reranking ({config.RERANK_MODEL})",
                "ms": (time.perf_counter() - t) * 1000,
                "detail": f"{len(candidates)} kept",
            })

        hits = apply_threshold(candidates)

        if not hits:
            return jsonify({
                "answer": NO_INFO,
                "sources": [],
                "context": [],
                "activity": {"steps": steps, "total_ms": (time.perf_counter() - overall) * 1000},
            })

        user_prompt = build_user_prompt(question, hits)

        # Stream from Groq server-side (still one blocking HTTP response to the
        # browser) so we can report first-token latency + token throughput.
        stats = {}
        for event, payload in generate_stream(user_prompt):
            if event == "done":
                stats = payload
        steps.append({
            "label": f"Waiting for LLM (Groq: {config.GEN_MODEL})",
            "ms": (stats.get("wall") or 0) * 1000,
            "llm": stats,
        })

        sources = list(dict.fromkeys(h["source"] for h in hits))
        return jsonify({
            "answer": stats.get("answer", ""),
            "sources": sources,
            "context": _context_data(hits),
            "prompt": {"system": SYSTEM_PROMPT, "user": user_prompt},
            "activity": {"steps": steps, "total_ms": (time.perf_counter() - overall) * 1000},
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/data")
def data_page():
    return render_template("data.html")


@app.route("/api/data", methods=["GET"])
def api_data():
    try:
        chunks = fetch_all()
        grouped = {}
        for c in chunks:
            grouped.setdefault(c["source"], []).append(c)
        return jsonify({"chunks_by_source": grouped, "total_chunks": len(chunks)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    try:
        from src.ingest import main as run_ingest
        run_ingest()
        return jsonify({"status": "ok", "message": "Ingestion complete."})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    """Save an uploaded .md/.txt/.pdf file into data/ (alongside the other documents).

    A PDF is converted to Markdown on arrival and it is that .md — not the PDF —
    that the rest of the pipeline sees. The original PDF stays in data/ for
    reference and re-conversion.

    Does not ingest — the user triggers /api/ingest_file separately. On the next
    full re-ingest the file is picked up automatically like any other in data/.
    """
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file provided."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    # Files prefixed `exclude-` are skipped by ingestion. If one is uploaded,
    # strip the prefix so it's saved as a normal, ingestable document.
    renamed_from = None
    if filename.startswith(config.EXCLUDE_PREFIX):
        stripped = filename[len(config.EXCLUDE_PREFIX):]
        if not Path(stripped).stem:
            return jsonify({
                "error": f"Nothing left after removing the '{config.EXCLUDE_PREFIX}' prefix from '{filename}'."
            }), 400
        renamed_from, filename = filename, stripped

    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        return jsonify({
            "error": f"Unsupported file type '{suffix}'. Allowed: .md, .txt, .pdf"
        }), 400

    dest = DATA_DIR / filename
    # `replaced` always describes the document that will be ingested — for a PDF
    # that's its Markdown twin, not the PDF itself.
    existed = (dest.with_suffix(".md") if suffix in PDF_SUFFIXES else dest).exists()
    file.save(dest)

    converted_from = None
    if suffix in PDF_SUFFIXES:
        try:
            md_path, _ = ensure_markdown(dest)
        except ValueError as exc:
            # Nothing usable came out of the PDF — don't leave it lying in data/
            # pretending to be an ingestable document.
            dest.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            dest.unlink(missing_ok=True)
            return jsonify({"error": f"Could not convert '{filename}': {exc}"}), 500
        # From here on the Markdown twin is the document.
        converted_from, filename = filename, md_path.name

    return jsonify({
        "status": "ok",
        "filename": filename,
        "replaced": existed,
        "renamed_from": renamed_from,
        "converted_from": converted_from,
    })


@app.route("/api/ingest_file", methods=["POST"])
def ingest_single():
    """Incrementally ingest one already-uploaded file from data/ into the index."""
    data = request.get_json(force=True)
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "filename is required."}), 400
    # Guard against path traversal: only operate on a bare filename in data/.
    if filename != secure_filename(filename):
        return jsonify({"error": "Invalid filename."}), 400

    try:
        result = ingest_file(filename)
        return jsonify({"status": "ok", **result})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    from src.store import get_index
    try:
        info = get_index().info()
        return jsonify({
            "index_name": config.INDEX_NAME,
            "num_chunks": info.get("num_docs", 0),
            "num_index_records": info.get("num_records", 0),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=True)
