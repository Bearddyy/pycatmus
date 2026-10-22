"""pycatmus command-line interface.

Mimics the original catmus.py CLI flag-for-flag:

  pycatmus PAGE.png OUT.txt
  pycatmus LINE.png OUT.txt --line
  pycatmus PAGE.png OUT.txt --red-filter --viz overlay.png
  pycatmus --batch "C:\\scans\\*.png" [--jobs N]
"""

from __future__ import annotations

import argparse
import glob as globmod
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__, core, models


def make_overlay(page: Path, hocr: str, out: Path) -> None:
    """Confidence overlay: word boxes coloured by confidence, per-line mean."""
    import re

    from PIL import Image, ImageDraw, ImageFont

    words = []
    for m in re.finditer(
        r'ocrx_word"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+); x_confs ([^";]+)', hocr
    ):
        x0, y0, x1, y1 = map(int, m.groups()[:4])
        confs = [float(c) for c in m.group(5).split() if c != "None"]
        tm = re.search(r">([^<]+)</span>", hocr[m.end() : m.end() + 200])
        words.append((x0, y0, x1, y1, confs, tm.group(1).strip() if tm else ""))

    line_boxes = [
        tuple(map(int, m.groups()))
        for m in re.finditer(r'ocr_line"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)', hocr)
    ]

    im = Image.open(page).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
    except OSError:
        font = None

    def color(c: float) -> tuple[int, int, int]:
        return (
            (255, 40, 40) if c < 0.6 else (255, 170, 0) if c < 0.85 else (40, 220, 80)
        )

    line_words: dict[int, list[float]] = {}
    for x0, y0, x1, y1, confs, _t in words:
        if not confs:
            continue
        c = sum(confs) / len(confs)
        ymid = (y0 + y1) / 2
        xmid = (x0 + x1) / 2
        dr.rectangle([x0, y0, x1, y1], outline=color(c), width=2)
        for li, (lx0, ly0, lx1, ly1) in enumerate(line_boxes):
            if ly0 <= ymid <= ly1 and lx0 <= xmid <= lx1:
                line_words.setdefault(li, []).append(c)
                break
    for li, (lx0, ly0, lx1, ly1) in enumerate(line_boxes):
        cs = line_words.get(li, [])
        s = sum(cs) / len(cs) if cs else -1
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
    ap = argparse.ArgumentParser(
        prog="pycatmus", description="CATMuS Medieval HTR transcription"
    )
    ap.add_argument("--version", action="version", version=f"pycatmus {__version__}")
    ap.add_argument("input", type=Path, nargs="?", help="page image (or use --batch)")
    ap.add_argument("output", type=Path, nargs="?", help="output .txt (or use --batch)")
    ap.add_argument(
        "--batch",
        metavar="GLOB",
        help='transcribe every image matching a glob, e.g. "C:\\scans\\*.png"; '
        "one .txt per image beside it (models load once for the whole run)",
    )
    ap.add_argument(
        "--line", action="store_true", help="input image is a single text line"
    )
    ap.add_argument(
        "--red-filter", action="store_true", help="remove red ink before OCR"
    )
    ap.add_argument(
        "--viz", type=Path, metavar="PNG", help="also write a confidence overlay image"
    )
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda:N (default auto)")
    ap.add_argument("--model", type=Path, help="recognition .mlmodel override")
    ap.add_argument("--seg-model", type=Path, help="segmentation .mlmodel override")
    ap.add_argument(
        "--keep-temp", action="store_true", help="keep intermediate red-filter images"
    )
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
    return ap


def run_single(args: argparse.ArgumentParser) -> None:
    if not (args.input and args.output):
        sys.exit("give INPUT OUTPUT, or use --batch GLOB")
    if not args.input.is_file():
        sys.exit(f"input image not found: {args.input}")
    core.preflight_write_check(args.output)

    engine = core.Engine(
        device=args.device, rec_model=args.model, seg_model=args.seg_model
    )
    tmp: Path | None = None
    if args.red_filter or args.viz:
        tmp = Path(tempfile.gettempdir()) / "pycatmus"
    result = engine.transcribe_page(
        args.input,
        single_line=args.line,
        red_ink=args.red_filter,
        want_hocr=bool(args.viz),
        scratch=tmp,
    )
    if args.viz:
        if result.hocr is None:  # pragma: no cover - want_hocr=True above
            sys.exit("internal error: no hOCR produced for overlay")
        page = args.input
        if args.red_filter and tmp is not None:
            page = tmp / (args.input.stem + "_pycatmus_nored.png")
        make_overlay(page, result.hocr, args.viz)
    args.output.write_text(result.text + "\n", encoding="utf-8")
    core.verify_output(args.output)
    if tmp is not None and not args.keep_temp:
        for junk in tmp.glob(f"{args.input.stem}_pycatmus_*"):
            junk.unlink(missing_ok=True)


def run_batch(args: argparse.ArgumentParser) -> None:
    if args.line:
        sys.exit("--batch works on page images, not single lines")
    pages = sorted(
        Path(p)
        for p in globmod.glob(str(args.batch))
        if Path(p).suffix.lower() in core.image_extensions()
    )
    if not pages:
        sys.exit(f"no images match: {args.batch}")
    core.preflight_write_check(pages[0].with_suffix(".txt"))

    if args.jobs > 1:
        _run_sharded(args, pages)
        return

    engine = core.Engine(
        device=args.device, rec_model=args.model, seg_model=args.seg_model
    )
    failures = 0
    for (result, err), page in zip(
        engine.transcribe_batch(
            pages,
            red_ink=args.red_filter,
            want_hocr=bool(args.viz),
            scratch=Path(tempfile.gettempdir()) / "pycatmus"
            if (args.red_filter or args.viz)
            else None,
        ),
        pages,
    ):
        if err is not None:
            failures += 1
            print(f"FAILED {page}: {err}", file=sys.stderr)
            continue
        assert result is not None
        if args.viz and result.hocr:
            src = page
            if args.red_filter:
                cand = (
                    Path(tempfile.gettempdir())
                    / "pycatmus"
                    / (page.stem + "_pycatmus_nored.png")
                )
                if cand.is_file():
                    src = cand
            make_overlay(src, result.hocr, page.with_suffix(".overlay.png"))
        out = page.with_suffix(".txt")
        out.write_text(result.text + "\n", encoding="utf-8")
        print(f"wrote {out.resolve()}")
    if failures:
        sys.exit(f"{failures} page(s) failed")


def _run_sharded(args: argparse.ArgumentParser, pages: list[Path]) -> None:
    """Round-robin sharding over N subprocesses (models load once per worker)."""
    procs: list[tuple[list[Path], subprocess.Popen[str]]] = []
    for shard in range(args.jobs):
        chunk = pages[shard :: args.jobs]
        if not chunk:
            continue
        manifest = Path(tempfile.gettempdir()) / f"pycatmus_shard_{shard}.lst"
        manifest.write_text("\n".join(str(p) for p in chunk), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "pycatmus",
            "--batch-list",
            str(manifest),
            "--device",
            args.device,
        ]
        if args.red_filter:
            cmd.append("--red-filter")
        if args.model:
            cmd += ["--model", str(args.model)]
        if args.seg_model:
            cmd += ["--seg-model", str(args.seg_model)]
        procs.append(
            (
                chunk,
                subprocess.Popen(
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
    for chunk, p in procs:
        err = p.stderr.read() if p.stderr else ""
        if err.strip():
            sys.stderr.write(err.rstrip() + "\n")
        if p.wait() != 0:
            failed = True
            print(f"worker failed on: {', '.join(p_.name for p_ in chunk)}", file=sys.stderr)
        else:
            for s in chunk:
                print(f"wrote {s.with_suffix('.txt').resolve()}")
    for manifest_done in (
        Path(tempfile.gettempdir()) / f"pycatmus_shard_{i}.lst" for i in range(args.jobs)
    ):
        manifest_done.unlink(missing_ok=True)
    if failed:
        sys.exit("one or more parallel workers failed")


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    ap.add_argument("--batch-list", type=Path, help=argparse.SUPPRESS)  # internal
    args = ap.parse_args(argv)

    if args.download_models:
        for name in models._MODELS:  # noqa: SLF001 - deliberate iteration
            models.download(name)
        return

    if args.batch_list:
        # shard worker: transcribe the pages named in a manifest file
        pages = [
            Path(ln)
            for ln in args.batch_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        engine = core.Engine(
            device=args.device, rec_model=args.model, seg_model=args.seg_model
        )
        failures = 0
        for (result, err), page in zip(
            engine.transcribe_batch(pages, red_ink=args.red_filter), pages
        ):
            if err is not None:
                failures += 1
                print(f"FAILED {page}: {err}", file=sys.stderr)
                continue
            assert result is not None
            page.with_suffix(".txt").write_text(result.text + "\n", encoding="utf-8")
        if failures:
            sys.exit(1)
        return

    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
