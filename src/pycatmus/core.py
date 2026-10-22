"""Transcription engine: in-process Kraken + CATMuS Medieval.

Mirrors what ``kraken segment | ocr`` does on the command line, using the
kraken 7 Python API so the package does not depend on a kraken.exe shim or a
particular install layout.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from . import models

_HOCR_WORD_RE = re.compile(
    r'ocrx_word"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)'
    r"(?:;\s*x_confs\s+([0-9.\s;]+?))?(?=[;\"])"
)
_HOCR_LINE_RE = re.compile(r'ocr_line"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)')

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def image_extensions() -> set[str]:
    return set(_IMAGE_EXTS)


@dataclass
class Word:
    text: str
    bbox: tuple[int, int, int, int]
    conf: float | None = None


@dataclass
class Line:
    bbox: tuple[int, int, int, int]
    words: list[Word] = field(default_factory=list)


@dataclass
class PageResult:
    source: Path
    lines: list[Line]
    hocr: str | None = None

    @property
    def text(self) -> str:
        """Graphematic text, one line per written line (never normalised)."""
        return "\n".join(" ".join(w.text for w in ln.words) for ln in self.lines)


def red_filter(src: Path, dst: Path) -> int:
    """Extract the green channel and paint red-ink pixels to parchment white.

    Dark ink is dark in every channel; red ink has a high red channel and a
    low green one; yellowish parchment is bright in green. So the green
    channel keeps black ink and background but fades red ink, and the mask
    step whitens what remains of it. Lossy by design: red that carries
    content is erased too.
    """
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    r, g = a[..., 0], a[..., 1]
    red_mask = (r - g > 40) & (r > 90)
    v = g.copy()
    v[red_mask] = 200
    Image.fromarray(v.clip(0, 255).astype(np.uint8)).save(dst)
    n = int(red_mask.sum())
    print(
        f"red-filter: {n} red pixels removed ({100 * red_mask.mean():.1f}% of page)",
        file=sys.stderr,
    )
    return n


def parse_hocr(html: str) -> list[Line]:
    """Parse kraken hOCR into lines/words with bounding boxes and confidences."""
    lines: list[Line] = []
    for lm in _HOCR_LINE_RE.finditer(html):
        lbbox = tuple(int(x) for x in lm.groups())
        assert lbbox is not None
        seg = html[lm.end() :]
        nxt = _HOCR_LINE_RE.search(seg)
        if nxt:
            seg = seg[: nxt.start()]
        words: list[Word] = []
        for wm in _HOCR_WORD_RE.finditer(seg):
            bbox = tuple(int(x) for x in wm.groups()[:4])
            conf_s = (wm.group(5) or "").strip().rstrip(";")
            conf: float | None = None
            if conf_s:
                vals = [float(c) for c in conf_s.split() if c != "None"]
                conf = sum(vals) / len(vals) if vals else None
            gt = seg.find(">", wm.end())
            end = seg.find("</span>", wm.end())
            text = seg[gt + 1 : end] if gt != -1 and end != -1 else ""
            text = (
                text.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
            )
            assert bbox is not None
            words.append(Word(text=text, bbox=bbox, conf=None if conf is None else round(conf, 3)))
        lines.append(Line(bbox=lbbox, words=[w for w in words if w.text.strip()]))
    return lines


class Engine:
    """One loaded pair of models; transcribe any number of pages."""

    def __init__(
        self,
        device: str = "auto",
        rec_model: Path | None = None,
        seg_model: Path | None = None,
        batch_size: int = 8,
    ) -> None:
        # torch must import before some numpy/torch interplay warnings settle;
        # imports stay lazy here so `pycatmus --help` is instant.
        from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

        self.device = self._normalise_device(device)
        rec_path = models.resolve("catmus-medieval", rec_model)
        seg_path = models.resolve("blla", seg_model)
        self.rec = RecognitionTaskModel.load_model(rec_path)
        self.seg = SegmentationTaskModel.load_model(seg_path)
        self.batch_size = batch_size

    @staticmethod
    def _normalise_device(device: str) -> dict[str, Any]:
        """Translate kraken-CLI-style device strings into Fabric config flags."""
        d = device.strip()
        if d in ("", "auto"):
            return {"accelerator": "auto", "device": "auto"}
        if d in ("cpu", "mps"):
            return {"accelerator": d, "device": "auto"}
        if d.startswith("cuda") or d == "gpu":
            dev, _, idx = d.partition(":")
            acc = "gpu" if dev.startswith("cuda") else "gpu"
            return {"accelerator": acc, "device": [int(idx)] if idx else "auto"}
        raise ValueError(f"invalid device {device!r} (use auto|cpu|mps|cuda:N)")

    def transcribe_page(
        self,
        page: Path,
        *,
        single_line: bool = False,
        red_ink: bool = False,
        want_hocr: bool = True,
        scratch: Path | None = None,
    ) -> PageResult:
        """Segment (unless *single_line*) and recognise one page image."""
        from kraken import serialization
        from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
        from kraken.containers import BBoxLine, Segmentation

        src = page
        if red_ink:
            work = scratch or page.parent
            work.mkdir(parents=True, exist_ok=True)
            filt = work / (page.stem + "_pycatmus_nored.png")
            red_filter(page, filt)
            src = filt

        im = Image.open(src)
        mode = im.mode
        if mode not in ("L", "RGB"):
            im = im.convert("L")

        if single_line:
            bounds = Segmentation(
                type="bbox",
                text_direction="horizontal-lr",
                imagename=str(src),
                script_detection=False,
                lines=[BBoxLine(id="_full", bbox=(0, 0, im.width, im.height))],
            )
        else:
            bounds = self.seg.predict(
                im,
                SegmentationInferenceConfig(
                    accelerator=self.device["accelerator"],
                    device=self.device["device"],
                    raise_on_error=True,
                ),
            )

        preds = list(
            self.rec.predict(
                im=im,
                segmentation=bounds,
                config=RecognitionInferenceConfig(
                    accelerator=self.device["accelerator"],
                    device=self.device["device"],
                    batch_size=self.batch_size,
                    raise_on_error=True,
                ),
            )
        )
        results = dataclasses.replace(bounds, lines=preds, imagename=str(src))

        hocr: str | None = None
        if want_hocr:
            hocr = serialization.serialize(
                results=results,
                image_size=im.size,
                writing_mode="horizontal-lr",
                scripts=None,
                template="hocr",
                template_source="native",
                processing_steps=[],
            )
            lines = parse_hocr(hocr)
        else:
            # plain-text fast path: no word boxes needed
            from kraken.containers import BaselineLine

            plain_lines: list[Line] = []
            for ln, r in zip(results.lines, preds):
                bbox: tuple[int, int, int, int]
                if isinstance(ln, BaselineLine):
                    bbox = ln.to_bbox().bbox
                else:
                    bbox = tuple(int(x) for x in ln.bbox)  # type: ignore[attr-defined]
                plain_lines.append(Line(bbox=bbox, words=[Word(text=r.prediction, bbox=bbox)]))
            lines = plain_lines
        return PageResult(source=page, lines=lines, hocr=hocr)

    def transcribe_batch(
        self, pages: list[Path], **kw: Any
    ) -> Iterator[tuple[PageResult | None, Exception | None]]:
        """Yield (result, error) per page; keep going on per-page failures."""
        for pg in pages:
            try:
                yield self.transcribe_page(pg, **kw), None
            except Exception as exc:  # noqa: BLE001 - deliberate per-page isolation
                yield None, exc


def iter_images(path: Path) -> list[Path]:
    """Images in *path* (a folder) or *[path]* if it is a file."""
    if path.is_dir():
        return sorted(
            q for q in path.iterdir() if q.suffix.lower() in _IMAGE_EXTS and q.is_file()
        )
    if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
        return [path]
    return []


def preflight_write_check(out: Path) -> None:
    """Fail before an expensive OCR run if the output dir is not writable."""
    target_dir = out.parent if str(out.parent) not in ("", ".") else Path.cwd()
    probe = target_dir / f".pycatmus_write_probe_{os.getpid()}"
    try:
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        sys.exit(f"cannot write to {target_dir}: {e}")


def verify_output(out: Path) -> None:
    """Read the output file back and prove the write persisted."""
    resolved = out.resolve()
    if not resolved.exists():
        sys.exit(f"ERROR: {resolved} does not exist after writing")
    size = resolved.stat().st_size
    if size == 0:
        sys.exit(f"ERROR: {resolved} exists but is empty")
    text = resolved.read_text(encoding="utf-8")
    n_lines = len([ln for ln in text.splitlines() if ln.strip()])
    print(f"wrote {resolved} ({size} bytes, {n_lines} non-empty lines)")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    print(f"read-back first line: {first}")
