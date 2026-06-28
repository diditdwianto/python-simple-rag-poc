"""Flask web UI for the RAG POC.

Run:  python -m src.app
"""

import time
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src import config
from src.embeddings import embed_query
from src.generate import generate_stream, SYSTEM_PROMPT
from src.query import NO_INFO, build_user_prompt
from src.store import apply_threshold, fetch_all, ping, search

load_dotenv()

app = Flask(__name__)


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

        t = time.perf_counter()
        qvec = embed_query(question)
        steps.append({"label": "Embedding query", "ms": (time.perf_counter() - t) * 1000})

        t = time.perf_counter()
        hits = apply_threshold(
            search(qvec, k=config.TOP_K, source=source, query_text=question)
        )
        steps.append({
            "label": f"Searching index ({config.SEARCH_MODE})",
            "ms": (time.perf_counter() - t) * 1000,
            "detail": f"{len(hits)} hits",
        })

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
