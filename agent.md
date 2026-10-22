# agent.md — operating instructions for this folder

HTR (handwritten text recognition) setup: Kraken 7 + CATMuS Medieval 1.6, Windows, RTX 4090.

## Run transcription

Always via the wrapper (it handles device, models, filters):

```bash
cd C:\Users\Byrne\kraken-catmus
.venv\Scripts\python.exe catmus.py PAGE.png OUT.txt
```

Flags: `--line` (input is a single line, no segmentation) · `--red-filter` (erase red ink: rubrics/strikethroughs) · `--viz overlay.png` (confidence overlay) · `--device cpu` · `--model`/`--seg-model` overrides.

## Parallelism

Batch mode scales across many parallel workers: `--batch GLOB --jobs N` splits
pages round-robin over N kraken processes. The workload is small-model,
low-GPU-load (CATMuS is ~16 MB; each worker uses roughly 2-3 GB VRAM counting
the CUDA context), so on this 24 GB RTX 4090 **large job counts are safe —
`--jobs 8` is comfortable, and more than that is possible if nothing else
needs the GPU**. Each worker pays its own ~12 s model warmup, so choose N so
that pages/N >> warmup (e.g. 40+ pages → `--jobs 8`). Separate `catmus.py`
invocations in parallel terminals also work fine; `--jobs` is just managed
sharding with per-worker error reporting. Do not run other GPU-heavy apps
(ComfyUI) alongside high `--jobs` counts without checking free VRAM first.

## Hard rules

- **Do not upgrade numpy to >= 2.5.** Kraken 7.1.1 calls `np.cross` on 2-D vectors; numpy 2.5 removed that support and lines silently vanish (symptom: most lines output empty, no errors). Fix: `uv pip install --python .venv\Scripts\python.exe "numpy<2.5"`.
- **Do not replace torch with a CPU build.** CUDA 12.6 build installed; GPU runs at `-d cuda:0` (the wrapper default).
- Transcription output is **raw/graphematic + NFD** — never "correct", expand, or normalise it unless the user asks; the user explicitly wants raw output.
- Red ink handling: use `--red-filter` when red = decoration/corrections; WITHOUT it when red carries content (it erases red).
- Page images work best flat, well-contrasted, ≥1500 px wide; cropped top/bottom lines transcribe badly — warn rather than "fix" silently.

## Files

- `catmus.py` — the CLI wrapper (red filter + kraken invocation + confidence overlay renderer).
- `app/app.py` + `app/static/index.html` — Flask web UI (start: `app\start.bat` or
  `.venv\Scripts\python.exe app\app.py` from this folder; serves http://127.0.0.1:5000).
  Loads image(s)/folder, transcribes via kraken hOCR (per-word bbox+confidence),
  hover-word-highlights-on-image, Export .txt. Flask installed in .venv.
- `catmus-medieval-1.6.0.mlmodel` — recognition model (Zenodo 21488839).
- `blla.mlmodel` — generic line segmentation model (Zenodo 14602568).
- `README.md` — full human docs (usage, pitfalls, model coverage).
- `.venv` — Python 3.12 + kraken 7.1.1 + torch cu126 + numpy 2.4.6.

## Debugging a bad result

1. Run with `--viz` and look at the overlay: blue underlines with no word boxes = segmentation/crop problem; red boxes = recogniser unsure.
2. Test the model alone on a hand-cropped line with `--line` — if that's good, the fault is segmentation, not recognition.
3. Raw kraken for diagnostics: `.venv\Scripts\kraken.exe -vv -r -d cuda:0 -i IN OUT segment -i blla.mlmodel ocr -m catmus-medieval-1.6.0.mlmodel` (check for `Extracting line failed`).
4. The `A NumPy version ...` UserWarning from scipy is harmless noise.
