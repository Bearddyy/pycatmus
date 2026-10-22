#!/usr/bin/env python
"""CATMuS Medieval HTR transcription CLI.

Usage:
  catmus.py PAGE.png OUT.txt            segment + transcribe a page
  catmus.py LINE.png OUT.txt --line     image is a single line (no segmentation)
  catmus.py PAGE.png OUT.txt --red-filter   drop red ink (rubrics/strikethroughs) first
  catmus.py PAGE.png OUT.txt --viz conf_overlay.png   write confidence overlay too
  catmus.py PAGE.png OUT.txt --keep-temp      keep intermediate images/JSON
Options:
  --device cuda:0|cpu   (default cuda:0)
  --model PATH          recognition model (default: catmus-medieval-1.6.0.mlmodel next to this script)
  --seg-model PATH      segmentation model   (default: blla.mlmodel next to this script)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def work_dir(page: Path) -> Path:
    return Path(tempfile.gettempdir()) / 'catmus' / (page.stem + '.catmus')


def red_filter(src: Path, dst: Path) -> None:
    """Extract the green channel and paint red-ink pixels to parchment white.

    Dark ink is dark in every channel; red ink has a high red channel and a
    low green one; yellowish parchment is bright in green. So the green
    channel keeps black ink and background but fades red ink, and the extra
    mask step whitens what remains of it.
    """
    from PIL import Image
    import numpy as np

    im = Image.open(src).convert('RGB')
    a = np.asarray(im).astype(np.int16)
    r, g = a[..., 0], a[..., 1]
    red_mask = (r - g > 40) & (r > 90)
    v = g.copy()
    v[red_mask] = 200
    Image.fromarray(v.clip(0, 255).astype(np.uint8)).save(dst)
    print(f'red-filter: {red_mask.sum()} red pixels removed '
          f'({100 * red_mask.mean():.1f}% of page)', file=sys.stderr)


def run_kraken(args, inputs: list[tuple[Path, Path]], extra: list[str]) -> None:
    kraken = str(HERE / '.venv' / 'Scripts' / 'kraken.exe')
    # -r / --raise-on-error: without it kraken swallows processing failures,
    # skips writing the output file, and still exits 0 (silent failure).
    cmd = [kraken, '-r', '-d', args.device]
    for src, dst in inputs:
        cmd += ['-i', str(src), str(dst)]
    cmd += extra
    # Force UTF-8 stdio in the child: kraken prints a '\u2713' progress mark
    # which crashes on Windows cp1252 consoles (PowerShell) with
    # UnicodeEncodeError *after* processing, silently losing the output file.
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding='utf-8', errors='replace', env=env)
    # relay kraken's stderr always: it logs per-line processing problems there
    if res.stderr.strip():
        sys.stderr.write(res.stderr.rstrip() + '\n')
    if res.returncode != 0:
        sys.stderr.write(res.stdout + '\n')
        sys.exit(f'kraken failed with exit code {res.returncode}')


def make_overlay(page: Path, hocr: Path, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    html = hocr.read_text(encoding='utf-8')
    words = []
    for m in re.finditer(
            r'ocrx_word"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+); x_confs ([^;"]+)',
            html):
        x0, y0, x1, y1 = map(int, m.groups()[:4])
        confs = [float(c) for c in m.group(5).split()]
        tm = re.search(r'>([^<]+)</span>', html[m.end():m.end() + 200])
        words.append((x0, y0, x1, y1, confs, tm.group(1).strip() if tm else ''))

    # line boxes from hOCR serve as the per-line bands
    line_boxes = []
    for m in re.finditer(r'ocr_line"[^>]*title="bbox (\d+) (\d+) (\d+) (\d+)',
                         html):
        line_boxes.append(tuple(map(int, m.groups())))

    im = Image.open(page).convert('RGB')
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 12)
    except OSError:
        font = None

    def color(c):
        return ((255, 40, 40) if c < .6 else
                (255, 170, 0) if c < .85 else (40, 220, 80))

    line_words = {}
    for x0, y0, x1, y1, confs, _t in words:
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
        dr.text((max(0, lx0 - 30), ly0 - 14),
                '-' if not cs else f'{s:.2f}', fill=col, font=font)
    im.save(out)
    print(f'overlay written: {out}', file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description='CATMuS Medieval HTR transcription')
    ap.add_argument('input', type=Path, nargs='?',
                    help='page image (or use --batch)')
    ap.add_argument('output', type=Path, nargs='?',
                    help='output .txt (or use --batch)')
    ap.add_argument('--batch', metavar='GLOB',
                    help='transcribe every image matching a glob, e.g. '
                         '"C:\\scans\\*.png"; one .txt per image beside it '
                         '(models load once for the whole batch)')
    ap.add_argument('--line', action='store_true',
                    help='input image is a single text line (skip segmentation)')
    ap.add_argument('--red-filter', action='store_true',
                    help='remove red ink (rubrics/strikethroughs) before OCR')
    ap.add_argument('--viz', type=Path, metavar='PNG',
                    help='also write a confidence overlay image (page mode)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--model', type=Path,
                    default=HERE / 'catmus-medieval-1.6.0.mlmodel')
    ap.add_argument('--seg-model', type=Path, default=HERE / 'blla.mlmodel')
    ap.add_argument('--keep-temp', action='store_true')
    ap.add_argument('--jobs', type=int, default=1, metavar='N',
                    help='with --batch: run N parallel kraken processes '
                         '(pages split round-robin, each loads models once; '
                         '~2 GB VRAM each — 4 is a good number on a 24 GB card)')
    args = ap.parse_args()

    if args.batch:
        if args.line:
            sys.exit('--batch works on page images, not single lines')
        import glob as globmod
        pages = sorted(Path(p) for p in globmod.glob(str(args.batch))
                       if Path(p).suffix.lower() in
                       ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        if not pages:
            sys.exit(f'no images match: {args.batch}')
        preflight_write_check(pages[0].with_suffix('.txt'))
        pairs: list[tuple[Path, Path]] = []
        for pg in pages:
            src = pg
            if args.red_filter:
                dst = work_dir(pg) / (pg.stem + '_nofilter.png')
                dst.parent.mkdir(parents=True, exist_ok=True)
                red_filter(pg, dst)
                src = dst
            pairs.append((src, pg.with_suffix('.txt')))

        ocr_args = ['segment', '-i', str(args.seg_model),
                    'ocr', '-B', '8', '-m', str(args.model)]
        if args.jobs > 1:
            kraken = str(HERE / '.venv' / 'Scripts' / 'kraken.exe')
            env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
            procs = []
            for shard in range(args.jobs):
                chunk = pairs[shard::args.jobs]
                if not chunk:
                    continue
                cmd = [kraken, '-r', '-d', args.device]
                for s, d in chunk:
                    cmd += ['-i', str(s), str(d)]
                cmd += ocr_args
                procs.append((chunk, subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True,
                    encoding='utf-8', errors='replace', env=env)))
            failed = False
            for chunk, p in procs:
                err = p.stderr.read()
                if err.strip():
                    sys.stderr.write(err.rstrip() + '\n')
                if p.wait() != 0:
                    failed = True
                    print(f'worker failed on: '
                          f'{", ".join(s.name for s, _ in chunk)}',
                          file=sys.stderr)
                else:
                    for _s, d in chunk:
                        print(f'wrote {d.resolve()}')
            if failed:
                sys.exit('one or more parallel kraken workers failed')
            return

        run_kraken(args, pairs, ocr_args)
        for _src, dst in pairs:
            print(f'wrote {dst.resolve()}')
        return

    if not (args.input and args.output):
        ap.error('give INPUT OUTPUT, or use --batch GLOB')

    work = work_dir(args.input)
    work.mkdir(parents=True, exist_ok=True)
    preflight_write_check(args.output)

    page = args.input
    if args.red_filter:
        page = work / (args.input.stem + '_nofilter.png')
        red_filter(args.input, page)

    if args.line:
        run_kraken(args, [(page, args.output)],
                   ['ocr', '-s', '-m', str(args.model)])
        verify_output(args.output)
        return

    if not args.seg_model.exists():
        sys.exit(f'segmentation model not found: {args.seg_model}\n'
                 f'download it with: .venv/Scripts/kraken.exe get 10.5281/zenodo.14602568 '
                 f'and copy blla.mlmodel next to this script')

    if args.viz:
        hocr = work / 'out.html'
        run_kraken(args, [(page, hocr)],
                   ['-h', 'segment', '-i', str(args.seg_model),
                    'ocr', '-m', str(args.model)])
        make_overlay(page, hocr, args.viz)
        run_kraken(args, [(page, args.output)],
                   ['segment', '-i', str(args.seg_model), 'ocr', '-m', str(args.model)])
    else:
        run_kraken(args, [(page, args.output)],
                   ['segment', '-i', str(args.seg_model), 'ocr', '-m', str(args.model)])
    verify_output(args.output)


def preflight_write_check(out: Path) -> None:
    """Fail before the expensive OCR run if we can't write the output file."""
    target_dir = out.parent.resolve() if out.parent != Path('') else Path.cwd()
    probe = target_dir / f'.catmus_write_probe_{os.getpid()}'
    try:
        probe.write_text('probe', encoding='utf-8')
        probe.unlink()
    except OSError as e:
        sys.exit(f'cannot write to {target_dir}: {e}\n'
                 f'check folder permissions (try as admin, or '
                 f'"Security" tab -> modify rights on the folder)')


def verify_output(out: Path) -> None:
    """Read the output file back and prove the write persisted."""
    import time

    resolved = out.resolve()
    deadline = time.time() + 5
    while time.time() < deadline:
        if resolved.exists():
            break
        time.sleep(0.25)
    if not resolved.exists():
        sys.exit(f'ERROR: {resolved} does not exist immediately after writing '
                 f'— the write was lost (check antivirus/sync tools, or the '
                 f'file was written to a different machine/container)')
    size = resolved.stat().st_size
    if size == 0:
        sys.exit(f'ERROR: {resolved} exists but is empty')
    text = resolved.read_text(encoding='utf-8')
    n_lines = len([l for l in text.splitlines() if l.strip()])
    print(f'wrote {resolved} ({size} bytes, {n_lines} non-empty lines)')
    first = next((l for l in text.splitlines() if l.strip()), '')
    print(f'read-back first line: {first}')


if __name__ == '__main__':
    main()
