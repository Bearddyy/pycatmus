"""PyCatmus web app: images left, editable transcription right, hover-to-locate.

Start with ``pycatmus-app`` (serves http://127.0.0.1:5000 by default).
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from . import core
from .core import Engine

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UPLOADS = Path(os.environ.get("PYCATMUS_UPLOADS", HERE / "uploads"))

app = Flask(__name__, static_folder=str(STATIC))

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
_ENGINE: Engine | None = None
ENGINE_LOCK = threading.Lock()


def get_engine(device: str) -> Engine:
    """One Engine per device string, built lazily and reused."""
    global _ENGINE
    with ENGINE_LOCK:
        if _ENGINE is None or getattr(_ENGINE, "_pycatmus_device", None) != device:
            _ENGINE = Engine(device=device)
            _ENGINE._pycatmus_device = device  # type: ignore[attr-defined]
        return _ENGINE


@app.route("/")
def index() -> Response:
    return send_from_directory(STATIC, "index.html")


@app.route("/api/load_local", methods=["POST"])
def load_local() -> Response:
    """Load every image from a local folder path, or a single image file."""
    body = request.get_json(silent=True) or {}
    p = Path(str(body.get("path", "")))
    files = core.iter_images(p)
    if not files:
        return jsonify(error=f"no images found at: {p}"), 400  # type: ignore[return-value]
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, f in enumerate(files):
        dest = job_dir / f"{i:03d}_{f.name}"
        dest.write_bytes(f.read_bytes())
        pages.append(
            {"id": f"{i:03d}", "name": f.name, "image_url": f"/api/image/{job_id}/{i:03d}"}
        )
    with JOBS_LOCK:
        JOBS[job_id] = {
            "pages": {pg["id"]: pg for pg in pages},
            "order": [pg["id"] for pg in pages],
            "status": "loaded",
        }
    return jsonify(job_id=job_id, pages=pages)  # type: ignore[return-value]


@app.route("/api/upload", methods=["POST"])
def upload() -> Response:
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="no files"), 400  # type: ignore[return-value]
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, fs in enumerate(files):
        name = os.path.basename(fs.filename or f"image{i}")
        if Path(name).suffix.lower() not in core.image_extensions():
            continue
        dest = job_dir / f"{i:03d}_{name}"
        fs.save(dest)
        pages.append(
            {"id": f"{i:03d}", "name": name, "image_url": f"/api/image/{job_id}/{i:03d}"}
        )
    if not pages:
        return jsonify(error="no supported images"), 400  # type: ignore[return-value]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "pages": {pg["id"]: pg for pg in pages},
            "order": [pg["id"] for pg in pages],
            "status": "loaded",
        }
    return jsonify(job_id=job_id, pages=pages)  # type: ignore[return-value]


def _orig_image(job_dir: Path, page_id: str) -> Path | None:
    hits = [
        q
        for q in sorted(job_dir.glob(f"{page_id}_*"))
        if "_nofilter" not in q.name and "_pycatmus_" not in q.name and ".shown." not in q.name
    ]
    return hits[0] if hits else None


@app.route("/api/image/<job_id>/<page_id>")
def image(job_id: str, page_id: str) -> Response:
    img = _orig_image(UPLOADS / job_id, page_id)
    if img is None:
        return jsonify(error="no such image"), 404  # type: ignore[return-value]
    if img.suffix.lower() in (".tif", ".tiff"):
        png = img.with_suffix(".shown.png")
        if not png.exists():
            from PIL import Image

            Image.open(img).convert("RGB").save(png)
        img = png
    return send_file(img)


@app.route("/api/generate", methods=["POST"])
def generate() -> Response:
    body = request.get_json(silent=True) or {}
    job_id = str(body.get("job_id", ""))
    red = bool(body.get("red_filter"))
    device = str(body.get("device", "auto"))
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404  # type: ignore[return-value]

    def worker() -> None:
        job["status"] = "running"
        job["error"] = None
        try:
            engine = get_engine(device)
            for pid in job["order"]:
                pg = job["pages"][pid]
                src = _orig_image(UPLOADS / job_id, pid)
                if src is None:
                    raise FileNotFoundError(f"page {pid} missing from uploads")
                result, err = next(
                    iter(
                        engine.transcribe_batch(
                            [src],
                            red_ink=red,
                            want_hocr=True,
                            scratch=UPLOADS / job_id,
                        )
                    )
                )
                if err is not None:
                    raise err
                assert result is not None
                pg["lines"] = [
                    {
                        "bbox": list(ln.bbox),
                        "words": [
                            {"text": w.text, "bbox": list(w.bbox), "conf": w.conf}
                            for w in ln.words
                        ],
                    }
                    for ln in result.lines
                ]
                pg["status"] = "done"
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(started=True)  # type: ignore[return-value]


@app.route("/api/status/<job_id>")
def status(job_id: str) -> Response:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404  # type: ignore[return-value]
    pages = [
        {
            "id": pid,
            "name": job["pages"][pid]["name"],
            "status": job["pages"][pid].get("status", "pending"),
            "lines": job["pages"][pid].get("lines"),
            "image_url": job["pages"][pid]["image_url"],
        }
        for pid in job["order"]
    ]
    return jsonify(status=job["status"], error=job.get("error"), pages=pages)  # type: ignore[return-value]


@app.route("/api/export", methods=["POST"])
def export() -> Response:
    """Plain .txt of the machine transcription (graphematic, never normalised)."""
    body = request.get_json(silent=True) or {}
    job_id = str(body.get("job_id", ""))
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404  # type: ignore[return-value]
    parts: list[str] = []
    for pid in job["order"]:
        for ln in job["pages"][pid].get("lines") or []:
            parts.append(" ".join(w["text"] for w in ln["words"]))
        parts.append("")  # blank line between pages
    text = "\n".join(parts).rstrip("\n") + "\n"
    out = UPLOADS / job_id / "transcription.txt"
    out.write_text(text, encoding="utf-8")
    return send_file(
        out,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="transcription.txt",
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="pycatmus-app", description="PyCatmus web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)
    UPLOADS.mkdir(exist_ok=True)
    print(f"PyCatmus web app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
