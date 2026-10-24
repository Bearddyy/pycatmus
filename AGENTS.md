# AGENTS.md — working notes for AI agents on PyCatmus

## What this is
pip-installable HTR package (src layout): CATMuS Medieval + Kraken 7 in-process,
CLI (`pycatmus`) and Flask web app (`pycatmus-app`, served from package data).
Quality bar: black (line 100), `mypy --strict` clean, pylint 10/10 (config in
pyproject). Keep all three green after any change.

## Environment
- venv: `.venv` (Python 3.12). CLI: `.venv/Scripts/pycatmus.exe`, app:
  `.venv/Scripts/pycatmus-app.exe` (port 5000).
- **numpy must stay <2.5** (Kraken 7.1.1 `np.cross` on 2-D vectors → lines
  silently vanish on 2.5+). Pinned in pyproject.
- Kraken device mapping: `--device cuda:0` → `accelerator="gpu", device=[0]`.
- Windows multiprocessing needs `if __name__ == "__main__":` guards.
- Models are NOT in git: auto-downloaded, md5-pinned, cached in
  `%LOCALAPPDATA%\pycatmus\models` (`PYCATMUS_MODELS` override).

## Image preparation lessons (hard-won on real pages)
- Line height in px is what matters, not page size: recogniser resizes line
  crops internally. Sweet spot 50–120 px; <35 px loses macrons/superscripts.
- Small/low-res inputs: upscale ×2–3 Lanczos (free win, segmenter starves
  below this). If BLLA throws `TopologyException: side location conflict` on
  an upscaled page, fall back to native scale + padding.
- Text touching image borders makes BLLA merge rows (observed 13 lines read
  as 71 fragments). Add ~20–25% white padding — fixes it, no pixels cropped.
- Never crop mid-line; edge-clipped lines are unrecoverable, flag them.
- Feed whole pages, not paragraph crops (segmenter needs layout context).
- JPEG q≥85 is fine; PNG better; artefacts mainly kill thin strokes.
- Training domain: CATMuS Medieval = 12th–15th c. Gothic Latin/French hands.
  It fails *confidently* outside: Anglo-Saxon square minuscule → garbage,
  17th-c court hand → garbage, 18th-c copperplate → fragments. McCATMuS
  (Zenodo 10.5281/zenodo.13788176) covers post-1500; `--model` takes it.
- hOCR parsing: word regex must use a lookahead for `x_confs` and slice text
  literally between `>` and `</span>` — consuming delimiters silently drops
  words (the old STANESTEDE bug).

## Transcription-delivery conventions (user's standing rules)
- **Never correct name tokens** (person or place): copy exactly as read,
  odd/garbled forms included; identifications are opinions and go in a
  clearly labelled appendix only. Same in translations — names carried over
  verbatim.
- Graphemic output: no abbreviation expansion in the text (use [ ]), no
  u/v or i/j normalisation; Tironian et = `⁊`.
- Always deliver the raw tool output as-is (RAW viewer export) alongside any
  corrected text — the user wants to see exactly what the tool produced.
- Don't crop user images to "improve" them; padding/upscaling only.
- Mark uncertain letters `[?]`, supplied text `<…>`; flag photo-clipped
  lines rather than inventing them.

## Web app / export
- `static/index.html` is the whole front end (vanilla JS, no build step).
  A literal `</script>` inside a JS template string closes the page's own
  script block early — escape as `<\/script>`.
- "Save HTML" builds a self-contained blob viewer: base64 images, edited
  text, canvas hover overlay. Edits typed by hand have no source bbox —
  those words highlight only the line band.
- Backend changes need a server restart; static/ changes just need a refresh.
- Headless verification pattern: drive http://127.0.0.1:5000 via Chrome CDP,
  override `window.download` to capture the blob, pixel-check the canvas
  after dispatching `mouseover` (see scratch export_*.py scripts in session
  history; canonical harness loads a folder, generates, exports, reopens).

## Verifying before commit
CLI single + `--line` + `--red-filter --viz` + `--batch --jobs` on GPU, web
load→generate→hover→export, then black/mypy/pylint, then wheel build
(`uv build -o dist`) — wheel must contain no model files.
