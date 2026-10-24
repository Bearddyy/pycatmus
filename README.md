PIP Package WIP.

# PyCatmus

Handwritten-text recognition (HTR) for medieval manuscripts, powered by the
[CATMuS Medieval](https://zenodo.org/records/21488839) model and the
[Kraken](https://kraken.re) engine — packaged as a command-line tool and a
local web app with word-level hover-to-source highlighting.

Covers Latin, Old & Middle French, Iberian languages, Italian, Middle Dutch
and Occitan Gothic hands (12th–15th century book and cursive scripts).
Transcription is **graphematic**: abbreviations stay exactly as written
(`uileꝰ`, not `villes`), nothing is normalised, expanded or spell-corrected.

## Install

```bash
pip install PyCatmus
```

This gives you a CPU-only install (torch comes from PyPI). For GPU
(NVIDIA, CUDA 12.6):

```bash
pip install PyCatmus[gpu] --extra-index-url https://download.pytorch.org/whl/cu126
```

The two model files (~21 MB total) are **not** bundled. They are downloaded
automatically from Zenodo on first use into a local cache and verified
against their recorded md5. To fetch them up front (e.g. on an offline
machine):

```bash
pycatmus --download-models
```

Cache location: `%LOCALAPPDATA%\pycatmus\models` on Windows,
`~/.local/share/pycatmus/models` on Linux (`$PYCATMUS_MODELS` overrides).
`--model` / `--seg-model` point at any other `.mlmodel` file.

## Command line

```bash
# whole page → txt
pycatmus folio1r.png folio1r.txt

# single hand-cropped line (skips page segmentation — most accurate isolate)
pycatmus line4.png line4.txt --line

# remove red ink (rubrics/strikethroughs) before OCR, and write a
# per-word confidence overlay
pycatmus folio1r.png out.txt --red-filter --viz overlay.png

# every page matching a glob, models load once, .txt beside each image
pycatmus --batch "C:\scans\*.png"

# batch 4 ways in parallel (each worker loads the models once, ~2 GB VRAM)
pycatmus --batch "C:\scans\*.png" --jobs 4

# device: auto (default) | cpu | cuda:N
pycatmus page.png out.txt --device cuda:0
```

| Flag | Effect |
|---|---|
| `--line` | input is one text line; skip segmentation |
| `--red-filter` | erase red ink before OCR (see *Red ink* below) |
| `--viz FILE.png` | confidence overlay: word boxes green/amber/red, per-line mean scores |
| `--batch GLOB` | transcribe all matching images; one `.txt` per image |
| `--jobs N` | parallel workers with `--batch` |
| `--device auto\|cpu\|cuda:N` | compute target (default `auto`) |
| `--model` / `--seg-model` | override recognition / segmentation model files |
| `--keep-temp` | keep intermediate red-filter images |
| `--download-models` | fetch both models from Zenodo and exit |

Output is raw graphematic Unicode in NFD form (diacritics and superscripts
are separate codepoints — some editors display spacing oddly; it is not
corrupted). Tironian *et* comes out as `⁊`.

Model warm-up (loading into GPU) costs ~10–15 s per process; for repeated
work prefer `--batch` over one call per page.

## Web app

```bash
pycatmus-app            # http://127.0.0.1:5000
pycatmus-app --port 8080
```

- **Load** a local folder or image path, or upload images. Multiple pages
  stack scrollably in the left pane.
- **Generate** transcribes every loaded page in one run (progress in the
  header; a *Remove red ink* checkbox applies the red filter to the batch).
- **Hover any word** to see its exact bounding box drawn on the page image,
  plus a band over its source line. Low-confidence words carry an amber
  (<0.85) or red (<0.6) underline — check those against the image.
- **Edit** the transcription in place: click a line and type. Enter moves to
  the next line instead of splitting, because source lineation is content.
- **Export .txt** saves what you see (your edits included, line breaks kept);
  **Export original .txt** saves the untouched machine output.
- **Save HTML** exports a single self-contained file (page images embedded,
  edited transcription, hover-to-locate intact) that opens in any browser —
  no Python, no network. Read-only snapshot, not a live editor.

Runs locally; nothing leaves your machine.

## Red ink

Medieval red ink and yellowish parchment both live in the image's RED channel:

- dark brown/black ink is dark in **every** channel
- red ink is dark in green/blue but bright in red
- yellow parchment is bright in green (its yellowness is excess red)

So the filter keeps only the **green channel** as greyscale (ink stays black,
parchment stays white) and paints detected red-ink pixels — where
`red − green > 40` — up to parchment white. Strikethroughs and rubrics vanish
instead of confusing the recogniser; on the sample page this fixes a word
that reads `Srauesteds` (garbled by a red deletion) into `STANESTEDE`.

The filter is **lossy by design**: a fully rubricated line will be erased
too. Use `--red-filter` when red = decoration or correction; leave it off
when red carries content you want transcribed.

## Reading the overlay (`--viz`)

- **Word boxes**: green ≥ 0.85 mean confidence, amber 0.6–0.85, red < 0.6
- **Underlines**: detected text lines; blue = line found but nothing recognised
- **Numbers**: mean confidence for that line (`-` = empty)

## Getting good results

**What the models actually see.** The recogniser crops each text line and
resizes it internally, so page size does not matter — **line height in
pixels is the currency**. Rules that came out of real pages:

- Aim for **50–120 px line height** in the image you feed. Below ~35 px
  superscripts, macrons and Tironian `⁊` dissolve; above ~150 px nothing
  improves (the recogniser downsamples anyway) — you only burn GPU.
- A full folio of ~35 lines ≈ **2,500–4,000 px wide**; archival scans at
  300–400 DPI sit in the sweet spot. 600 DPI adds nothing.
- **Small images (phone photos, resized attachments) rescue well with
  ×2–3 Lanczos upscaling.** Harmless to the recogniser, and often decisive
  for the segmenter.
- **White padding fixes shattered segmentation.** A page whose text touches
  the image borders makes BLLA glue neighbouring rows together (observed:
  ~20 lines → 71 fragments). Adding a ~20–25 % white band around the image
  restored 13/13 correct lines on the same photo. Padding is not a crop —
  no pixels are lost.
- **Never crop mid-line.** Lines sliced by the photo edge transcribe badly
  and the missing part is unrecoverable; flag them instead.
- Whole pages beat paragraph crops: the segmenter uses layout context
  (baseline rhythm, margins) that a tight crop removes. If you must crop,
  crop generously with margin and add padding.
- **JPEG is fine** (doc-quality ≥ 85): artefacts only hurt when they eat
  thin marks — macrons, superscripts, hairline strokes. PNG is strictly
  better if you have it; never re-save a JPEG through several generations,
  and transcoding JPEG→PNG recovers nothing.

**Know the model's training domain.** CATMuS Medieval reads Gothic Latin/
French-era hands; it produces confident garbage on anything outside:
Anglo-Saxon square minuscule fails completely, 17th-c. court hand fails,
18th-c. copperplate shatters. For post-1500 English hands use McCATMuS
(`--model`, Zenodo 10.5281/zenodo.13788176); for anything else, treat the
raw output as a layout guide and verify per line with `--line`.

Also:

- Flat, well-contrasted scans, ideally ≥1500 px wide. Museum photos of
  angled books degrade badly.
- For a stubborn word, crop the line tightly and run with `--line`.
- Check the `--viz` overlay to find low-confidence words before reading.

## Model notes

- CATMuS Medieval is one multilingual model — there is **no language flag**.
- Best on 12th–15th c. Gothic book hands and Gothic cursive/administrative
  hands.
- The Kraken model repository has newer/related models: `kraken list`,
  `kraken get DOI`. McCATMuS (Zenodo 10.5281/zenodo.13788176) covers
  16th-century to modern hands — point `--model` at it for later documents.

## Provenance and licences

| Component | Licence |
|---|---|
| PyCatmus code | MIT |
| CATMuS Medieval 1.6 model | [CC-BY-4.0](https://zenodo.org/records/21488839) |
| BLLA segmentation model | Apache-2.0 (Zenodo 10.5281/zenodo.14602568) |
| Kraken | Apache-2.0 |

If you publish transcriptions made with this tool, please credit the CATMuS
project (CREMMA, GalliCorpora, HTRomance, DEEDS; curator Thibault Clérice)
via the model DOI, and cite Kraken (Kiessling, *ICDAR 2025*,
doi:10.1007/978-3-032-04624-6_26). Transcription *output* itself carries no
licence from any of these — what you may publish depends on the terms
attached to the **source images**, not the software.

## Development

```bash
git clone <this repo> && cd pycatmus
uv venv .venv
uv pip install -e . "numpy<2.5" torch          # + --torch-backend=cu126 for GPU
```

Pitfalls:

- **numpy must stay `<2.5`** — Kraken 7.1.1's line-crop geometry calls
  `np.cross` on 2-D vectors; numpy 2.5 removed that support and text lines
  silently vanish (empty lines, no error). The pin is declared in
  `pyproject.toml`.
- On Windows consoles, child processes printing non-ASCII need
  `PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8` (handled internally).
- Scipy prints a numpy-version `UserWarning` on every run; harmless noise.
- BLLA can raise `TopologyException: side location conflict` on some
  upscaled inputs (invalid predicted polygons). Observed on a 2–3× enlarged
  copperplate page; retry at native scale + white padding instead — do not
  treat it as corrupt input.
