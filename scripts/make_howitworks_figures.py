"""Generate 'how it works' figures from a red-filtered run on a colour page.

Pipeline mirrors what the engine actually does:
  colour original -> x3 Lanczos upscale + 15 % white pad (prep page)
  prep page       -> red_filter (green channel, red ink painted white)
  red-filtered    -> BLLA segmentation + CATMuS recognition (model input)
  boxes/conf      -> transposed BACK onto the colour original for the crops

Produces into docs/img/:
  how1-redfilter.png      colour vs red-filtered side-by-side (what the model sees)
  how1b-segmentation.png  kraken geometry over the red-filtered page
  how2-line-crops.png     per-line crops cut from the COLOUR page at boxes
                          generated on the red-filtered image, recognised text beside
  how3-word-crop.png      one word lifted from colour at its filtered-run bbox

Usage:
  PYTHONPATH=src python scripts/make_howitworks_figures.py <colour-page> [word-substring]

NOTE: run as a file, never piped to stdin (kraken spawns children; Windows
cannot re-import '<stdin>' and forks in an infinite crash loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "img"
SCRATCH = REPO / ".figscratch"
UP = 3.0
NAVY = (15, 23, 42)
GREY = (148, 163, 184)


def prep_page(orig: Path) -> tuple[Path, int]:
    """x3 Lanczos upscale + 15 % white padding; returns (prep png, pad px)."""
    SCRATCH.mkdir(exist_ok=True)
    im = Image.open(orig).convert("RGB")
    w, h = int(im.width * UP), int(im.height * UP)
    pad = int(0.15 * max(w, h))
    up = im.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (255, 255, 255))
    canvas.paste(up, (pad, pad))
    out = SCRATCH / (orig.stem + "_prep.png")
    canvas.save(out)
    return out, pad


def main() -> None:
    orig_path = Path(sys.argv[1])
    word_keys = [k.lower() for k in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["rannulf"])]

    from pycatmus.core import Engine, red_filter

    prep, pad = prep_page(orig_path)
    filtered = SCRATCH / (orig_path.stem + "_nored.png")
    n_red = red_filter(prep, filtered)
    print(f"red-filter removed {n_red} px")

    eng = Engine()  # device auto
    res = eng.transcribe_page(filtered, want_hocr=True)

    # curved geometry pass on the exact model input (red-filtered page)
    from kraken.configs import SegmentationInferenceConfig

    im0 = Image.open(filtered)
    if im0.mode not in ("L", "RGB"):
        im0 = im0.convert("L")
    dev = dict(eng.device)
    dev["raise_on_error"] = True
    bounds = eng.seg.predict(im0, SegmentationInferenceConfig(**dev))
    geom_lines = list(bounds.lines)

    def outer_ring(boundary):
        if not boundary:
            return []
        return boundary if isinstance(boundary[0], (list, tuple)) else []

    orig = Image.open(orig_path).convert("RGB")

    def crop_orig(bbox):
        """Crop the colour original using bbox coords in preprocessed space."""
        x0, y0, x1, y1 = bbox
        ox0 = max(0, int((x0 - pad) / UP))
        oy0 = max(0, int((y0 - pad) / UP))
        ox1 = min(orig.width, int((x1 - pad) / UP))
        oy1 = min(orig.height, int((y1 - pad) / UP))
        return orig.crop((ox0, oy0, ox1, oy1))

    prep_im = Image.open(prep).convert("RGB")
    scale = 1600 / prep_im.width

    # ---- how1: colour vs red-filtered, side by side -----------------------
    filt_im = Image.open(filtered)
    filt_rgb = filt_im.convert("RGB")
    if filt_rgb.mode != "RGB":  # red_filter writes greyscale
        filt_rgb = Image.merge("RGB", (filt_im, filt_im, filt_im))
    sw = int(prep_im.width * scale)
    sh = int(prep_im.height * scale)
    left = prep_im.resize((sw, sh), Image.Resampling.LANCZOS)
    right = filt_rgb.resize((sw, sh), Image.Resampling.LANCZOS)
    h1 = Image.new("RGB", (sw * 2 + 30, sh + 40), NAVY)
    d = ImageDraw.Draw(h1)
    h1.paste(left, (10, 30))
    h1.paste(right, (sw + 20, 30))
    d.text((10, 8), "input (colour, upscaled+padded):", fill=GREY)
    d.text((sw + 20, 8), "what the model reads (red ink erased):", fill=GREY)
    h1.save(OUT / "how1-redfilter.png")

    # ---- how1b: segmentation over the red-filtered page -------------------
    f1b = right.copy()
    d = ImageDraw.Draw(f1b)
    for reg in getattr(bounds, "regions", []):
        try:
            pts = [(int(x * scale), int(y * scale)) for x, y in outer_ring(reg.coordinates)]
            d.line(pts + pts[:1], fill=(167, 139, 250), width=2, joint="curve")
        except Exception:
            pass
    for i, line in enumerate(geom_lines):
        bl = getattr(line, "baseline", None)
        ring = outer_ring(getattr(line, "boundary", None))
        if ring:
            pts = [(int(x * scale), int(y * scale)) for x, y in ring]
            d.line(pts + pts[:1], fill=(52, 211, 153), width=1, joint="curve")
        if bl is not None and len(bl) >= 2:
            pts = [(int(x * scale), int(y * scale)) for x, y in bl]
            d.line(pts, fill=(251, 113, 133), width=2, joint="curve")
            d.text((pts[0][0], max(0, pts[0][1] - 12)), str(i), fill=(251, 191, 36))
        else:
            bb = getattr(line, "bbox", None) or line.to_bbox().bbox
            x0, y0, x1, y1 = (int(v * scale) for v in bb)
            d.rectangle([x0, y0, x1, y1], outline=(52, 211, 153), width=2)
            d.text((x0 + 2, max(0, y0 - 11)), str(i), fill=(251, 191, 36))
    f1b.save(OUT / "how1b-segmentation.png")

    # ---- how2: line crops cut ALONG the segmentation polygon ---------------
    # each crop is pasted through its boundary polygon as a mask: the navy
    # canvas shows through outside the contour — no parchment rectangles.
    import numpy as np

    crops = []
    for idx, line in enumerate(res.lines):
        txt = " ".join(w.text for w in line.words).strip()
        if not txt:
            continue
        x0, y0, x1, y1 = line.bbox
        if (y1 - y0) < 25:  # drop degenerate margin-mark 'lines'
            continue
        geom = outer_ring(getattr(geom_lines[idx], "boundary", None)) if idx < len(geom_lines) else None
        crop = crop_orig((x0, y0, x1, y1)).copy()
        ox0 = max(0, int((x0 - pad) / UP))
        oy0 = max(0, int((y0 - pad) / UP))
        mask = None
        pts: list[tuple[float, float]] = []
        if geom:
            pts = [((x - pad) / UP - ox0, (y - pad) / UP - oy0) for x, y in geom]
            m = Image.new("L", crop.size, 0)
            ImageDraw.Draw(m).polygon(pts, fill=255)
            mask = m
        crop = crop.resize(
            (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
            Image.Resampling.LANCZOS,
        )
        if mask is not None:
            mask = mask.resize(crop.size, Image.Resampling.LANCZOS)
            arr = np.asarray(crop).copy()
            arr[np.asarray(mask) < 128] = NAVY
            crop = Image.fromarray(arr)
            # redraw contour AFTER masking so the outline is not clipped
            rx = crop.width / max(1, int((x1 - x0)))
            ry = crop.height / max(1, int((y1 - y0)))
            ImageDraw.Draw(crop).line(
                [(px * rx, py * ry) for px, py in pts] + [(pts[0][0] * rx, pts[0][1] * ry)],
                fill=(220, 38, 38),
                width=2,
                joint="curve",
            )
        crops.append((crop, txt))
    strip_w = sw + 380
    strip_h = sum(c.height + 30 for c, _ in crops) + 10 * len(crops)
    f2 = Image.new("RGB", (strip_w, strip_h), NAVY)
    d = ImageDraw.Draw(f2)
    y = 5
    for crop, txt in crops:
        f2.paste(crop, (5, y))
        d.text((crop.width + 15, y + crop.height // 2), txt[:120], fill=(226, 232, 240))
        y += crop.height + 20
    f2.save(OUT / "how2-line-crops.png")

    # ---- how3: one word from the colour page -------------------------------
    target_word, target_line = None, None
    best_conf = -1.0
    for key in word_keys:
        for line in res.lines:
            for w in line.words:
                t = w.text.strip().lower()
                if len(t) > 4 and key in t and w.conf > best_conf:
                    target_word, target_line, best_conf = w, line, w.conf
        if target_word is not None:
            break
    assert target_word is not None, f"none of {word_keys} found on this page"

    lx0, ly0, lx1, ly1 = target_line.bbox
    line_img = crop_orig((lx0, ly0, lx1, ly1)).resize(
        (int((lx1 - lx0) * scale), int((ly1 - ly0) * scale)), Image.Resampling.LANCZOS
    )
    wx0, wy0, wx1, wy1 = target_word.bbox
    # small margin so descenders/final letters are not clipped
    wx0, wy0 = wx0 - 10, wy0 - 10
    wx1, wy1 = wx1 + 10, wy1 + 10

    # cut the word with the SAME shape as the line crops: the line's curved
    # boundary polygon (clipped horizontally by the word window), navy outside
    cx = (target_line.bbox[0] + target_line.bbox[2]) / 2
    cy = (target_line.bbox[1] + target_line.bbox[3]) / 2

    def _centre(gl):
        bb = gl.bbox if getattr(gl, "bbox", None) else gl.to_bbox().bbox
        return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)

    ring: list = []
    best_d = 1e18
    for gl in geom_lines:
        gx, gy = _centre(gl)
        dist = (gx - cx) ** 2 + (gy - cy) ** 2
        if dist < best_d:
            best_d, r = dist, outer_ring(getattr(gl, "boundary", None))
            if r:
                ring = r
    word_img = crop_orig((wx0, wy0, wx1, wy1)).copy()
    if ring:
        import numpy as np

        ox0 = max(0, int((wx0 - pad) / UP))
        oy0 = max(0, int((wy0 - pad) / UP))
        pts = [((x - pad) / UP - ox0, (y - pad) / UP - oy0) for x, y in ring]
        m = Image.new("L", word_img.size, 0)
        ImageDraw.Draw(m).polygon(pts, fill=255)
        arr = np.asarray(word_img).copy()
        arr[np.asarray(m) < 128] = NAVY
        word_img = Image.fromarray(arr)
        ImageDraw.Draw(word_img).polygon(pts, outline=(220, 38, 38))
    else:
        ImageDraw.Draw(word_img).rectangle(
            [0, 0, word_img.width - 1, word_img.height - 1], outline=(220, 38, 38), width=2
        )
    z = max(1, min(620 // max(1, word_img.width), 5))
    word_img = word_img.resize((word_img.width * z, word_img.height * z), Image.Resampling.LANCZOS)

    w_canvas = max(line_img.width, 900)
    f3 = Image.new("RGB", (w_canvas, line_img.height + word_img.height + 90), NAVY)
    d = ImageDraw.Draw(f3)
    d.text((5, 5), "parent line (colour page, box from red-filtered run):", fill=GREY)
    f3.paste(line_img, (5, 25))
    px = int((wx0 - lx0) * scale)
    pw = int((wx1 - wx0) * scale)
    d.rectangle([5 + px, 25, 5 + px + pw, 25 + line_img.height], outline=(251, 191, 36), width=2)
    d.text((5, 40 + line_img.height), "that word, lifted from the colour page:", fill=GREY)
    f3.paste(word_img, (5, 60 + line_img.height))
    d.text(
        (15, 80 + line_img.height + word_img.height),
        f'"{target_word.text.strip()}"  conf={target_word.conf:.3f}',
        fill=(52, 211, 153),
    )
    f3.save(OUT / "how3-word-crop.png")

    print("wrote how1, how1b, how2, how3 to", OUT)


if __name__ == "__main__":
    main()
