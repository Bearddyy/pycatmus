# CATMuS Medieval HTR — local setup

Handwritten-text recognition for medieval manuscripts (Latin, Old/Middle French,
Spanish, Italian) using **Kraken 7** + the **CATMuS Medieval 1.6** model, on
this machine's RTX 4090.

- Folder: `C:\Users\Byrne\kraken-catmus`
- Model: `catmus-medieval-1.6.0.mlmodel` (Zenodo record 21488839, v1.6.2, CC-BY-4.0,
  md5 `326a6ea675661ec9ce0bb6b1958de495` — verified)
- Segmentation: `blla.mlmodel` (generic print+handwriting line segmenter,
  Zenodo 14602568, copied into this folder)

## Usage

Everything goes through `catmus.py`:

```bash
cd C:\Users\Byrne\kraken-catmus
.venv\Scripts\python.exe catmus.py PAGE.png OUT.txt
```

| Option | What it does |
|---|---|
| `--red-filter` | Removes red ink (rubrics, decorated initials, red strikethroughs) before OCR. See below. |
| `--line` | Treat the input image as ONE text line; skips page segmentation. Best accuracy when you crop a line yourself. |
| `--viz FILE.png` | Also write a confidence overlay: word boxes coloured by confidence, per-line mean score. |
| `--device cpu` | Use CPU instead of the GPU. |
| `--model / --seg-model` | Point at different .mlmodel files. |
| `--keep-temp` | Keep the intermediate `*.catmus` folder next to the input. |

Examples:

```bash
# whole page
.venv\Scripts\python.exe catmus.py folio1r.png folio1r.txt

# page + red-ink removal + visual confidence check
.venv\Scripts\python.exe catmus.py folio1r.png folio1r.txt --red-filter --viz folio1r_overlay.png

# one hand-cropped line
.venv\Scripts\python.exe catmus.py line4.png line4.txt --line

# batch: every page in a folder, models load ONCE for the whole run,
# one .txt written beside each image (--red-filter/--device work too)
.venv\Scripts\python.exe catmus.py --batch "C:\scans\*.png"

# batch 4 ways in parallel: N kraken workers, pages split round-robin.
# Each worker holds its own model copy (~2 GB VRAM) and its own ~12 s
# warmup, so --jobs N pays off when pages/N >> warmup, i.e. larger sets.
.venv\Scripts\python.exe catmus.py --batch "C:\scans\*.png" --jobs 4
```

Model warmup (load into GPU) costs ~10-15 s per process launch; for repeated
work use `--batch` rather than one call per page.

## The red filter

Medieval red ink and yellowish parchment both live in the image's RED channel:

- dark brown/black ink is dark in **every** channel
- red ink is dark in green/blue but bright in red
- yellow parchment is bright in green (its yellowness is excess red)

So the filter keeps only the **green channel** as greyscale (ink stays black,
parchment stays white) and then **paints detected red-ink pixels** — where
`red − green > 40` — up to parchment white. Strikethroughs and rubrics vanish
instead of confusing the recogniser; on the test page this fixed a word that
read `Srauesteds` (garbled by a red deletion) into `STANESTEDE`.

The filter is lossy by design: red text (a full rubricated line) will be erased
too. Use `--red-filter` when red is decoration/correction, omit it when red
carries content you want transcribed.

## Reading the overlay

- **Word boxes**: green ≥ 0.85 mean confidence, amber 0.6–0.85, red < 0.6
- **Underlines**: detected text lines; blue = line found but nothing recognised
- **Numbers**: mean confidence for that line (`-` = empty)

Amber/red words are the ones to check against the image.

## Output conventions

- Transcription is **graphematic**: abbreviations are transcribed as written,
  never expanded (`uileꝰ` not `villes`).
- Output is **NFD-normalised** Unicode: diacritics/superscripts are separate
  codepoints, so some editors display spacing oddly. It's not corrupted, just
  decomposed.
- Tironian et comes out as `⁊`; the cropped top/bottom lines of a page
  transcribe badly — crop them out or ignore them.

## Raw Kraken (what catmus.py wraps)

```bash
.venv\Scripts\kraken.exe -d cuda:0 -i page.png out.txt ^
    segment -i blla.mlmodel ocr -m catmus-medieval-1.6.0.mlmodel
.venv\Scripts\kraken.exe -d cuda:0 -i line.png out.txt ocr -s -m catmus-medieval-1.6.0.mlmodel
```

Add `-h` before the subcommands to get hOCR (per-word confidence + boxes).

## Environment / known pitfalls

- Python 3.12 venv at `.venv`, managed with `uv`.
- **numpy must stay < 2.5**: Kraken 7.1.1's line-crop geometry calls
  `np.cross` on 2-D vectors, which numpy 2.5 removed — the symptom is pages
  where only a few lines transcribe and the rest come back empty with no
  error. If you ever see that, run:
  `uv pip install --python .venv\Scripts\python.exe "numpy<2.5"`
- torch is the CUDA 12.6 build (2.14.1+cu126); reinstall with
  `uv pip install --python .venv\Scripts\python.exe --torch-backend=cu126 --force-reinstall torch torchvision`
  if it breaks.
- Scipy prints a numpy-version UserWarning on every run; harmless noise.

## Model notes

- CATMuS Medieval covers: Old & Middle French (best), Latin, Spanish/other
  Iberian, Italian, Middle Dutch, Occitan. Strictly graphemic, NFD-trained.
- It is one multilingual model — there is **no language flag** to set.
- Best on 12th–15th c. Gothic book hands and Gothic cursive/administrative
  hands; scans should be flat, well-contrasted, ideally ≥ 1500 px wide.
- Repository also has newer/related models: `kraken.exe list`, download with
  `kraken.exe get DOI` (models land in `%LOCALAPPDATA%\htrmopo`).
  McCATMuS (zenodo 13788176) covers 16th c.–modern for later documents.
