#!/usr/bin/env python
"""CATMuS Medieval HTR web app.

Run from the kraken-catmus folder:
    .venv\\Scripts\\python.exe app\\app.py
then open http://127.0.0.1:5000

Pages are uploaded (or loaded from a local folder), transcribed with
Kraken + CATMuS Medieval via hOCR so every word carries a bounding box,
and the front-end shows image left / text right with hover-to-highlight.
"""
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

HERE = Path(__file__).resolve().parent          # .../kraken-catmus/app
ROOT = HERE.parent                              # .../kraken-catmus
KRAKEN = ROOT / '.venv' / 'Scripts' / 'kraken.exe'
REC_MODEL = ROOT / 'catmus-medieval-1.6.0.mlmodel'
SEG_MODEL = ROOT / 'blla.mlmodel'
UPLOADS = HERE / 'uploads'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

app = Flask(__name__, static_folder='static')

# job_id -> {'pages': {page_id: {...}}, 'status': ..., 'text': ..., 'error': ...}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
OCR_SEMAPHORE = threading.Semaphore(1)   # one kraken batch at a time

WORD_RE = re.compile(
    r'ocrx_word"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)'
    r'(?:;\s*x_confs\s+([0-9.\s;]+?))?(?=[;"])')
LINE_RE = re.compile(r'ocr_line"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)')


def parse_hocr(html: str):
    """Return [{text, bbox:[x0,y0,x1,y1], conf}, ...] per line, in order."""
    lines = []
    for lm in LINE_RE.finditer(html):
        lx0, ly0, lx1, ly1 = map(int, lm.groups())
        # words inside this line span only
        seg = html[lm.end():]
        nxt = LINE_RE.search(seg)
        if nxt:
            seg = seg[:nxt.start()]
        words = []
        for wm in WORD_RE.finditer(seg):
            x0, y0, x1, y1 = map(int, wm.groups()[:4])
            confs = (wm.group(5) or '').strip().rstrip(';')
            conf = None
            if confs:
                vals = [float(c) for c in confs.split() if c not in ('None',)]
                conf = sum(vals) / len(vals) if vals else None
            gt = seg.find('>', wm.end())
            end = seg.find('</span>', wm.end())
            text = seg[gt + 1:end] if gt != -1 and end != -1 else ''
            # hOCR text is HTML-escaped
            text = (text.replace('&amp;', '&').replace('&lt;', '<')
                    .replace('&gt;', '>').replace('&quot;', '"')
                    .replace('&#39;', "'"))
            words.append({'text': text,
                          'bbox': [x0, y0, x1, y1],
                          'conf': round(conf, 3) if conf is not None else None})
        words = [w for w in words if w['text'].strip()]
        lines.append({'bbox': [lx0, ly0, lx1, ly1], 'words': words})
    return lines


def red_filter(src: Path, dst: Path) -> int:
    """Extract the green channel and paint red-ink pixels to parchment white
    (same algorithm as the catmus.py CLI --red-filter)."""
    from PIL import Image
    import numpy as np
    im = Image.open(src).convert('RGB')
    a = np.asarray(im).astype(np.int16)
    r, g = a[..., 0], a[..., 1]
    red_mask = (r - g > 40) & (r > 90)
    v = g.copy()
    v[red_mask] = 200
    Image.fromarray(v.clip(0, 255).astype(np.uint8)).save(dst)
    return int(red_mask.sum())


def run_kraken_hocr(src: Path, dst: Path):
    cmd = [str(KRAKEN), '-r', '-d', 'cuda:0', '-h',
           '-i', str(src), str(dst),
           'segment', '-i', str(SEG_MODEL),
           'ocr', '-m', str(REC_MODEL)]
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding='utf-8', errors='replace', env=env,
                         cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout)[-2000:])
    return dst.read_text(encoding='utf-8')


@app.route('/')
def index():
    return send_from_directory(HERE / 'static', 'index.html')


@app.route('/api/load_local', methods=['POST'])
def load_local():
    """Load every image from a local folder path, or a single image file."""
    p = Path(request.json.get('path', ''))
    files = []
    if p.is_dir():
        files = sorted(q for q in p.iterdir()
                       if q.suffix.lower() in IMAGE_EXTS and q.is_file())
    elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        files = [p]
    else:
        return jsonify(error=f'not a folder or image: {p}'), 400
    if not files:
        return jsonify(error=f'no images found in {p}'), 400
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, f in enumerate(files):
        dest = job_dir / f'{i:03d}_{f.name}'
        dest.write_bytes(f.read_bytes())
        pages.append({'id': f'{i:03d}', 'name': f.name,
                      'image_url': f'/api/image/{job_id}/{i:03d}'})
    with JOBS_LOCK:
        JOBS[job_id] = {'pages': {pg['id']: dict(pg) for pg in pages},
                        'order': [pg['id'] for pg in pages],
                        'status': 'loaded'}
    return jsonify(job_id=job_id, pages=pages)


@app.route('/api/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files:
        return jsonify(error='no files'), 400
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, fs in enumerate(files):
        name = os.path.basename(fs.filename or f'image{i}')
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        dest = job_dir / f'{i:03d}_{name}'
        fs.save(dest)
        pages.append({'id': f'{i:03d}', 'name': name,
                      'image_url': f'/api/image/{job_id}/{i:03d}'})
    if not pages:
        return jsonify(error='no supported images'), 400
    with JOBS_LOCK:
        JOBS[job_id] = {'pages': {pg['id']: dict(pg) for pg in pages},
                        'order': [pg['id'] for pg in pages],
                        'status': 'loaded'}
    return jsonify(job_id=job_id, pages=pages)


@app.route('/api/image/<job_id>/<page_id>')
def image(job_id, page_id):
    job_dir = UPLOADS / job_id
    hits = [q for q in sorted(job_dir.glob(f'{page_id}_*'))
            if '_nofilter' not in q.name]
    if not hits:
        return jsonify(error='no such image'), 404
    img = hits[0]
    if img.suffix.lower() in ('.tif', '.tiff'):
        png = img.with_suffix('.shown.png')
        if not png.exists():
            from PIL import Image
            Image.open(img).convert('RGB').save(png)
        img = png
    resp = send_file(img)
    # tif may not render in-browser; convert if needed
    return resp


@app.route('/api/generate', methods=['POST'])
def generate():
    job_id = request.json.get('job_id')
    red = bool(request.json.get('red_filter'))
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(error='unknown job'), 404

    def worker():
        job['status'] = 'running'
        job['error'] = None
        try:
            with OCR_SEMAPHORE:
                for pid in job['order']:
                    pg = job['pages'][pid]
                    src = next(q for q in sorted((UPLOADS / job_id).glob(f'{pid}_*'))
                               if '_nofilter' not in q.name and '.shown.' not in q.name)
                    out = (UPLOADS / job_id / f'{pid}.hocr')
                    ocr_src = src
                    if red:
                        ocr_src = (UPLOADS / job_id / f'{pid}_nofilter.png')
                        red_filter(src, ocr_src)
                    if src.suffix.lower() in ('.tif', '.tiff', '.webp'):
                        # kraken handles tif, browsers mostly don't — make a png
                        from PIL import Image
                        png = src.with_suffix('.shown.png')
                        if not png.exists():
                            Image.open(src).convert('RGB').save(png)
                    html = run_kraken_hocr(ocr_src, out)
                    pg['lines'] = parse_hocr(html)
                    pg['status'] = 'done'
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify(started=True)


@app.route('/api/status/<job_id>')
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(error='unknown job'), 404
    pages = []
    for pid in job['order']:
        pg = job['pages'][pid]
        pages.append({'id': pid, 'name': pg['name'],
                      'status': pg.get('status', 'pending'),
                      'lines': pg.get('lines'),
                      'image_url': pg['image_url']})
    return jsonify(status=job['status'], error=job.get('error'), pages=pages)


@app.route('/api/export', methods=['POST'])
def export():
    """Build plain .txt (graphematic, one line per written line)."""
    job_id = request.json.get('job_id')
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error='unknown job'), 404
    parts = []
    for pid in job['order']:
        pg = job['pages'][pid]
        for ln in pg.get('lines') or []:
            parts.append(' '.join(w['text'] for w in ln['words']))
        parts.append('')  # blank line between pages
    text = '\n'.join(parts).rstrip('\n') + '\n'
    out = UPLOADS / job_id / 'transcription.txt'
    out.write_text(text, encoding='utf-8')  # NFD raw — never normalise
    return send_file(out, mimetype='text/plain; charset=utf-8',
                     as_attachment=True, download_name='transcription.txt')


if __name__ == '__main__':
    UPLOADS.mkdir(exist_ok=True)
    if not KRAKEN.exists():
        sys.exit(f'kraken not found: {KRAKEN}\nrun this app from the kraken-catmus install')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
