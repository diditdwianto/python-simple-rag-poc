"""Flask web UI for the RAG POC.

Run:  python -m src.app
"""

import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src import config
from src.generate import generate, SYSTEM_PROMPT
from src.query import NO_INFO, build_user_prompt, retrieve
from src.store import fetch_all

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    source = data.get("source") or None

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        hits = retrieve(question, source=source)
        if not hits:
            return jsonify({"answer": NO_INFO, "sources": [], "context": []})

        user_prompt = build_user_prompt(question, hits)
        answer = generate(user_prompt)

        sources = list(dict.fromkeys(h["source"] for h in hits))
        context_data = [
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

        return jsonify({
            "answer": answer,
            "sources": sources,
            "context": context_data,
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
            },
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
