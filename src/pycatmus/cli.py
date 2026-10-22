"""pycatmus command-line interface.

Mimics the original catmus.py CLI flag-for-flag::

    pycatmus PAGE.png OUT.txt
    pycatmus LINE.png OUT.txt --line
    pycatmus PAGE.png OUT.txt --red-filter --viz overlay.png
    pycatmus --batch "C:\\scans\\*.png" [--jobs N]
"""

from __future__ import annotations

import argparse
import glob as globmod
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import __version__, models
from .core import Engine, image_extensions, preflight_write_check, verify_output


@dataclass
class Options:
    """Parsed CLI options."""

    input: Path | None = None
    output: Path | None = None
    batch: str | None = None
    batch_list: Path | None = None
    line: bool = False
    red_filter: bool = False
    viz: Path | None = None
    device: str = "auto"
    model: Path | None = None
    seg_model: Path | None = None
    keep_temp: bool = False
    jobs: int = 1
    download_models: bool = False


_TMP = Path(tempfile.gettempdir()) / "pycatmus"


def make_overlay(page: Path, hocr: str, out: Path) -> None:
    """Confidence overlay: word boxes coloured by confidence, per-line mean."""
    words: list[tuple[int, int, int, int, list[float]]] = []
    for m in re.finditer(
        r'ocrx_word"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+); x_confs ([^";]+)', hocr
    ):
        confs = [float(c) for c in m.group(5).split() if c != "None"]
        words.append(
            (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                confs,
            )
        )

    line_boxes: list[tuple[int, int, int, int]] = [
        (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        )
        for m in re.finditer(r'ocr_line"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)', hocr)
    ]

    im = Image.open(page).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font: ImageFont.FreeTypeFont | None = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
    except OSError:
        font = None

    def color(c: float) -> tuple[int, int, int]:
        if c < 0.6:
            return (255, 40, 40)
        if c < 0.85:
            return (255, 170, 0)
        return (40, 220, 80)

    line_words: dict[int, list[float]] = {}
    for x0, y0, x1, y1, confs in words:
        if not confs:
            continue
        c = sum(confs) / len(confs)
        ymid = (y0 + y1) / 2
        xmid = (x0 + x1) / 2
        dr.rectangle((x0, y0, x1, y1), outline=color(c), width=2)
        for li, (lx0, ly0, lx1, ly1) in enumerate(line_boxes):
            if ly0 <= ymid <= ly1 and lx0 <= xmid <= lx1:
                line_words.setdefault(li, []).append(c)
                break
    for li, (lx0, ly0, lx1, ly1) in enumerate(line_boxes):
        cs = line_words.get(li, [])
        s = sum(cs) / len(cs) if cs else -1.0
        col = (0, 140, 255) if not cs else color(s)
        dr.line([lx0, ly1 - 2, lx1, ly1 - 2], fill=col, width=1)
        dr.text(
            (max(0, lx0 - 30), ly0 - 14),
            "-" if not cs else f"{s:.2f}",
            fill=col,
            font=font,
        )
    im.save(out)
    print(f"overlay written: {out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ``pycatmus`` console script."""
    ap = argparse.ArgumentParser(prog="pycatmus", description="CATMuS Medieval HTR transcription")
    ap.add_argument("--version", action="version", version=f"pycatmus {__version__}")
    ap.add_argument("input", type=Path, nargs="?", help="page image (or use --batch)")
    ap.add_argument("output", type=Path, nargs="?", help="output .txt (or use --batch)")
    ap.add_argument(
        "--batch",
        metavar="GLOB",
        help='transcribe every image matching a glob, e.g. "C:\\scans\\*.png"; '
        "one .txt per image beside it (models load once for the whole run)",
    )
    ap.add_argument("--line", action="store_true", help="input image is a single text line")
    ap.add_argument("--red-filter", action="store_true", help="remove red ink before OCR")
    ap.add_argument("--viz", type=Path, metavar="PNG", help="also write a confidence overlay image")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda:N (default auto)")
    ap.add_argument("--model", type=Path, help="recognition .mlmodel override")
    ap.add_argument("--seg-model", type=Path, help="segmentation .mlmodel override")
    ap.add_argument("--keep-temp", action="store_true", help="keep intermediate red-filter images")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="with --batch: run N parallel pycatmus processes "
        "(pages split round-robin; each loads the models once)",
    )
    ap.add_argument(
        "--download-models",
        action="store_true",
        help="fetch the CATMuS Medieval and BLLA models from Zenodo and exit",
    )
    ap.add_argument("--batch-list", type=Path, help=argparse.SUPPRESS)  # internal
    return ap


def to_options(ns: argparse.Namespace) -> Options:
    """Convert argparse output to a typed Options record."""
    return Options(
        input=ns.input,
        output=ns.output,
        batch=ns.batch,
        batch_list=ns.batch_list,
        line=ns.line,
        red_filter=ns.red_filter,
        viz=ns.viz,
        device=ns.device,
        model=ns.model,
        seg_model=ns.seg_model,
        keep_temp=ns.keep_temp,
        jobs=ns.jobs,
        download_models=ns.download_models,
    )


def _engine(opts: Options) -> Engine:
    return Engine(device=opts.device, rec_model=opts.model, seg_model=opts.seg_model)


def _scratch(opts: Options) -> Path | None:
    return _TMP if (opts.red_filter or opts.viz) else None


def _overlay_source(page: Path, opts: Options) -> Path:
    """The page image the OCR actually consumed (filtered copy if red-filtered)."""
    if opts.red_filter:
        cand = _TMP / (page.stem + "_pycatmus_nored.png")
        if cand.is_file():
            return cand
    return page


def run_single(opts: Options) -> None:
    """Transcribe one page image to one .txt (plus optional overlay)."""
    if opts.input is None or opts.output is None:
        sys.exit("give INPUT OUTPUT, or use --batch GLOB")
    if not opts.input.is_file():
        sys.exit(f"input image not found: {opts.input}")
    preflight_write_check(opts.output)

    result = _engine(opts).transcribe_page(
        opts.input,
        single_line=opts.line,
        red_ink=opts.red_filter,
        want_hocr=bool(opts.viz),
        scratch=_scratch(opts),
    )
    if opts.viz is not None:
        if result.hocr is None:  # pragma: no cover - want_hocr=True above
            sys.exit("internal error: no hOCR produced for overlay")
        make_overlay(_overlay_source(opts.input, opts), result.hocr, opts.viz)
    opts.output.write_text(result.text + "\n", encoding="utf-8")
    verify_output(opts.output)
    if _scratch(opts) is not None and not opts.keep_temp:
        for junk in _TMP.glob(f"{opts.input.stem}_pycatmus_*"):
            junk.unlink(missing_ok=True)


def _write_page_result(page: Path, text: str, hocr: str | None, opts: Options) -> None:
    if opts.viz and hocr:
        make_overlay(_overlay_source(page, opts), hocr, page.with_suffix(".overlay.png"))
    out = page.with_suffix(".txt")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out.resolve()}")


def run_batch(opts: Options) -> None:
    """Transcribe every image matching a glob, models loaded once."""
    if opts.line:
        sys.exit("--batch works on page images, not single lines")
    if opts.batch is None:
        sys.exit("--batch GLOB required")
    pages = sorted(
        Path(p) for p in globmod.glob(opts.batch) if Path(p).suffix.lower() in image_extensions()
    )
    if not pages:
        sys.exit(f"no images match: {opts.batch}")
    preflight_write_check(pages[0].with_suffix(".txt"))

    if opts.jobs > 1:
        run_sharded(opts, pages)
        return

    failures = 0
    for (result, err), page in zip(
        _engine(opts).transcribe_batch(
            pages,
            red_ink=opts.red_filter,
            want_hocr=bool(opts.viz),
            scratch=_scratch(opts),
        ),
        pages,
    ):
        if err is not None or result is None:
            failures += 1
            print(f"FAILED {page}: {err}", file=sys.stderr)
            continue
        _write_page_result(page, result.text, result.hocr, opts)
    if failures:
        sys.exit(f"{failures} page(s) failed")


def run_sharded(opts: Options, pages: list[Path]) -> None:
    """Round-robin sharding over N subprocesses (models load once per worker)."""
    procs: list[tuple[list[Path], subprocess.Popen[str]]] = []
    manifests: list[Path] = []
    for shard in range(opts.jobs):
        chunk = pages[shard :: opts.jobs]
        if not chunk:
            continue
        manifest = Path(tempfile.gettempdir()) / f"pycatmus_shard_{shard}.lst"
        manifest.write_text("\n".join(str(p) for p in chunk), encoding="utf-8")
        manifests.append(manifest)
        cmd = [
            sys.executable,
            "-m",
            "pycatmus",
            "--batch-list",
            str(manifest),
            "--device",
            opts.device,
        ]
        if opts.red_filter:
            cmd.append("--red-filter")
        if opts.model:
            cmd += ["--model", str(opts.model)]
        if opts.seg_model:
            cmd += ["--seg-model", str(opts.seg_model)]
        procs.append(
            (
                chunk,
                subprocess.Popen(  # pylint: disable=consider-using-with
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ),
            )
        )
    failed = False
    for chunk, proc in procs:
        err = proc.stderr.read() if proc.stderr else ""
        if err.strip():
            sys.stderr.write(err.rstrip() + "\n")
        if proc.wait() != 0:
            failed = True
            names = ", ".join(p.name for p in chunk)
            print(f"worker failed on: {names}", file=sys.stderr)
        else:
            for p in chunk:
                print(f"wrote {p.with_suffix('.txt').resolve()}")
    for manifest in manifests:
        manifest.unlink(missing_ok=True)
    if failed:
        sys.exit("one or more parallel workers failed")


def run_shard_worker(opts: Options) -> None:
    """Worker mode: transcribe the pages listed in a --batch-list manifest."""
    assert opts.batch_list is not None
    pages = [
        Path(ln) for ln in opts.batch_list.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    failures = 0
    for (result, err), page in zip(
        _engine(opts).transcribe_batch(pages, red_ink=opts.red_filter, want_hocr=False),
        pages,
    ):
        if err is not None or result is None:
            failures += 1
            print(f"FAILED {page}: {err}", file=sys.stderr)
            continue
        page.with_suffix(".txt").write_text(result.text + "\n", encoding="utf-8")
    if failures:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``pycatmus`` console script."""
    opts = to_options(build_parser().parse_args(argv))

    if opts.download_models:
        for name in models.KNOWN_MODELS:
            models.download(name)
        return
    if opts.batch_list is not None:
        run_shard_worker(opts)
        return
    if opts.batch is not None:
        run_batch(opts)
    else:
        run_single(opts)


if __name__ == "__main__":
    main()
