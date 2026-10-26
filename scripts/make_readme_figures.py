"""Generate README 'How it works' figures from a real transcription run.

Produces into docs/img/:
  fig1-segmentation.png  detected lines over the page (bbox + index)
  fig2-line-crops.png    stacked per-line crops with the recognised text
  fig3-word-crop.png     one word lifted out, with its parent line above it

Usage:
  PYTHONPATH=src python scripts/make_readme_figures.py <page.png> <model.mlmodel> [word-substring]

NOTE: run as a file, never piped to stdin (kraken spawns children; Windows
cannot re-import '<stdin>' and forks in an infinite crash loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pycatmus.core import Engine

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"


def main() -> None:
    page, model = Path(sys.argv[1]), Path(sys.argv[2])
    word_key = sys.argv[3].lower() if len(sys.argv) > 3 else "don"
    # optional: colour original + the upscale factor used to make <page>;
    # crops are taken from the ORIGINAL at mapped coordinates (padding is the
    # standard int(0.15*max(upsize)) recipe from the README)
    orig_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    upfactor = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    OUT.mkdir(parents=True, exist_ok=True)

    eng = Engine(rec_model=model)
    res = eng.transcribe_page(page)

    # second pass to keep kraken's curved geometry (region polygons + baselines);
    # cheap: model already loaded, and hOCR bboxes above stay used for text/conf
    from kraken.configs import SegmentationInferenceConfig
    from PIL import Image as _Image

    im0 = _Image.open(page)
    if im0.mode not in ("L", "RGB"):
        im0 = im0.convert("L")
    dev = dict(eng.device)
    dev["raise_on_error"] = True
    bounds = eng.seg.predict(im0, SegmentationInferenceConfig(**dev))

    im = Image.open(page).convert("RGB")
    scale = 1600 / im.width
    small = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)

    # colour ORIGINAL the page was derived from (upscale xN + pad recipe), so
    # crops show the real parchment rather than the grey preprocessed input
    orig = Image.open(orig_path).convert("RGB") if orig_path else None
    pad = 0
    sx = sy = 1.0
    if orig is not None:
        # page size = (orig*w + 2*pad, orig*h + 2*pad), pad = int(0.15*orig_max)
        ow, oh = orig.width, orig.height
        p = int(0.15 * max(ow * upfactor, oh * upfactor))
        assert abs(ow * upfactor + 2 * p - im.width) <= 2 and abs(
            oh * upfactor + 2 * p - im.height
        ) <= 2, "original/upscaled-padded page geometry mismatch"
        pad = p
        sx = sy = upfactor

    def crop_orig(bbox):
        """Crop the colour original using bbox coords in preprocessed space."""
        x0, y0, x1, y1 = bbox
        if orig is None:
            return im.crop((x0, y0, x1, y1)), (1.0, 1.0)
        ox0 = max(0, int((x0 - pad) / sx))
        oy0 = max(0, int((y0 - pad) / sy))
        ox1 = min(orig.width, int((x1 - pad) / sx))
        oy1 = min(orig.height, int((y1 - pad) / sy))
        return orig.crop((ox0, oy0, ox1, oy1)), (1.0 / sx, 1.0 / sy)

    # ---- fig 1: segmentation overlay (true kraken geometry) ---------------
    def outer_ring(boundary):
        # kraken geometry = plain [[x, y], [x, y], ...] point lists
        if not boundary:
            return []
        return boundary if isinstance(boundary[0], (list, tuple)) else []

    f1 = small.copy()
    d = ImageDraw.Draw(f1)
    # region outlines (usually MainZone polygons)
    for reg in getattr(bounds, "regions", []):
        try:
            pts = [(int(x * scale), int(y * scale)) for x, y in outer_ring(reg.coordinates)]
            d.line(pts + pts[:1], fill=(167, 139, 250, 255), width=2, joint="curve")
        except Exception:
            pass
    for i, line in enumerate(bounds.lines):
        bl = getattr(line, "baseline", None)
        ring = outer_ring(getattr(line, "boundary", None))
        if ring:
            pts = [(int(x * scale), int(y * scale)) for x, y in ring]
            d.line(pts + pts[:1], fill=(52, 211, 153), width=1, joint="curve")
        if bl is not None and len(bl) >= 2:
            pts = [(int(x * scale), int(y * scale)) for x, y in bl]
            d.line(pts, fill=(251, 113, 133), width=2, joint="curve")
            d.text((pts[0][0], max(0, pts[0][1] - 12)), str(i), fill=(251, 191, 36))
        else:  # BBoxLine fallback
            bb = getattr(line, "bbox", None) or line.to_bbox().bbox
            x0, y0, x1, y1 = (int(v * scale) for v in bb)
            d.rectangle([x0, y0, x1, y1], outline=(52, 211, 153), width=2)
            d.text((x0 + 2, max(0, y0 - 11)), str(i), fill=(251, 191, 36))
    f1.save(OUT / "fig1-segmentation.png")
    # ---- fig 2: per-line crops with recognised text ----------------------
    geom_lines = list(bounds.lines)
    crops = []
    for idx, line in enumerate(res.lines):
        txt = " ".join(w.text for w in line.words).strip()
        if not txt:
            continue
        x0, y0, x1, y1 = line.bbox
        geom = outer_ring(getattr(geom_lines[idx], "boundary", None)) if idx < len(geom_lines) else None
        crop, (msx, msy) = crop_orig((x0, y0, x1, y1))
        crop = crop.copy()
        if geom:
            # outline of the exact cut-out polygon, mapped into crop-local coords
            cd = ImageDraw.Draw(crop)
            pts = [
                (max(0, int((x - pad) * msx)) - max(0, int((x0 - pad) * msx)),
                 max(0, int((y - pad) * msy)) - max(0, int((y0 - pad) * msy)))
                for x, y in geom
            ]
            cd.line(pts + pts[:1], fill=(220, 38, 38), width=2, joint="curve")
        # normalise to the figure's display scale
        crop = crop.resize(
            (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
            Image.Resampling.LANCZOS,
        )
        crops.append((crop, txt))
    strip_w = int(im.width * scale) + 340
    strip_h = sum(c.height + 30 for c, _ in crops) + 10 * len(crops)
    f2 = Image.new("RGB", (strip_w, strip_h), (15, 23, 42))
    d = ImageDraw.Draw(f2)
    y = 5
    for crop, txt in crops:
        f2.paste(crop, (5, y))
        d.rectangle([5, y, 5 + crop.width, y + crop.height], outline=(71, 85, 105))
        d.text((crop.width + 15, y + crop.height // 2), txt[:110], fill=(226, 232, 240))
        y += crop.height + 20
    f2.save(OUT / "fig2-line-crops.png")

    # ---- fig 3: one word lifted out --------------------------------------
    target_word, target_line = None, None
    for line in res.lines:
        for w in line.words:
            t = w.text.strip().lower()
            if len(t) > 4 and word_key in t:
                target_word, target_line = w, line
    assert target_word is not None, f"demo word '{word_key}' not found on this page"

    lx0, ly0, lx1, ly1 = target_line.bbox
    line_img, (msx, msy) = crop_orig((lx0, ly0, lx1, ly1))
    line_img = line_img.resize(
        (int((lx1 - lx0) * scale), int((ly1 - ly0) * scale)), Image.Resampling.LANCZOS
    )
    wx0, wy0, wx1, wy1 = target_word.bbox
    word_img, _ = crop_orig((wx0, wy0, wx1, wy1))
    word_img = word_img.copy()
    wd = ImageDraw.Draw(word_img)
    wd.rectangle([0, 0, word_img.width - 1, word_img.height - 1], outline=(220, 38, 38), width=2)
    word_img = word_img.resize(
        (min(word_img.width * 3, 620), word_img.height * 3), Image.Resampling.LANCZOS
    )

    w = max(line_img.width, 900)
    f3 = Image.new("RGB", (w, line_img.height + word_img.height + 90), (15, 23, 42))
    d = ImageDraw.Draw(f3)
    d.text((5, 5), "parent line as recognised:", fill=(148, 163, 184))
    f3.paste(line_img, (5, 25))
    px = int((wx0 - lx0) * scale)
    pw = int((wx1 - wx0) * scale)
    d.rectangle([5 + px, 25, 5 + px + pw, 25 + line_img.height], outline=(251, 191, 36), width=2)
    d.text((5, 40 + line_img.height), "that word, cropped by its bbox:", fill=(148, 163, 184))
    f3.paste(word_img, (5, 60 + line_img.height))
    d.text(
        (15, 80 + line_img.height + word_img.height),
        f'"{target_word.text.strip()}"  conf={target_word.conf}',
        fill=(52, 211, 153),
    )
    f3.save(OUT / "fig3-word-crop.png")

    # ---- fig 4: one line cut ALONG its curved polygon (from colour orig) --
    # match the geom line whose bbox centre is nearest the demo line's centre
    cx = (target_line.bbox[0] + target_line.bbox[2]) / 2
    cy = (target_line.bbox[1] + target_line.bbox[3]) / 2

    def centre(gl):
        bb = gl.bbox if getattr(gl, "bbox", None) else gl.to_bbox().bbox
        return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)

    di, best = None, 1e18
    for i, gl in enumerate(geom_lines):
        gx, gy = centre(gl)
        dist = (gx - cx) ** 2 + (gy - cy) ** 2
        if dist < best:
            best, di = dist, i
    ring = outer_ring(getattr(geom_lines[di], "boundary", None)) if di is not None else []
    if ring:
        lbbox = [min(p[0] for p in ring), min(p[1] for p in ring),
                 max(p[0] for p in ring), max(p[1] for p in ring)]
        crop, _ = crop_orig(lbbox)
        crop = crop.copy()
        # white out everything outside the polygon -> true curved cut-out
        ox0 = max(0, int((lbbox[0] - pad) / sx))
        oy0 = max(0, int((lbbox[1] - pad) / sy))
        m = Image.new("L", crop.size, 0)
        md = ImageDraw.Draw(m)
        pts = [((x - pad) / sx - ox0, (y - pad) / sy - oy0) for x, y in ring]
        md.polygon(pts, fill=255)
        import numpy as np

        arr = np.asarray(crop).copy()
        arr[np.asarray(m) < 128] = (255, 255, 255)
        crop = Image.fromarray(arr)
        # 3x zoom for visibility on the navy canvas
        f4 = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (f4.width + 20, f4.height + 50), (15, 23, 42))
        canvas.paste(f4, (10, 30))
        cd = ImageDraw.Draw(canvas)
        cd.text(
            (10, 8),
            "one line cut along its curved boundary polygon (colour original):",
            fill=(148, 163, 184),
        )
        canvas.save(OUT / "fig4-curved-cutout.png")
        print("wrote fig4 to", OUT)

    print("wrote fig1, fig2, fig3 to", OUT)


if __name__ == "__main__":
    main()
