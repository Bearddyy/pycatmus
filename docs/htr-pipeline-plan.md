# HTR Pipeline Plan — Manuscript Transcription with pycatmus

Companion to `htr-pipeline.html` (the diagrams). Everything below is grounded in
live tests run on this machine, Oct 2026.

## The one-paragraph version

A document goes in; Hermes classifies its **script family, date and language(s)**
and splits mixed pages (e.g. a Latin charter with Old English bounds) into
per-language chunks. Each chunk is preprocessed (upscale, padding, optional red
filter) and routed to the recognition model that fits its domain; the shared
kraken engine + BLLA segmenter do the actual reading. Outputs are re-stitched
into one transcription with confidence data. Low-confidence pages get corrected
by a human in pycatmus-app — and those corrections are the raw material for the
training loop that makes each lane's model better over time.

## Stage 1 — Intake & AI triage

Input: any scan/photo of a manuscript page. Hermes inspects it and decides:

| Signals looked at | Decision |
|---|---|
| Date formulae, personal names, document type | period band |
| Letterforms: `þ ð ƿ ꝺ ꞇ` present? looped ascenders? secretary `e/re`? | script family |
| Language(s) on page; are they in separable regions? | routing chunks |

**Multilingual split:** a page like the Æthelred charter we tested is one file
but two recognition problems. The Latin grant body and the vernacular boundary
clause are segmented into separate images (region-level crops, generous
margins — never mid-line), each routed independently, outputs re-stitched in
page order at the end.

**Gate:** confident classifications auto-route; anything exotic (copperplate,
Anglo-Norman charter hand, unknown) is flagged to the user rather than
silently fed to a wrong model — the failure mode we saw is *confident garbage*,
which is worse than an honest error message.

## Stage 2 — Routing table (what exists vs what must be built)

| Lane | Domain | Model | Status |
|---|---|---|---|
| A | Gothic book/chancery hands ≤1500 (Lat/Fr/It/Iber/Dutch/Occ) | **CATMuS Medieval 1.6** | ✅ installed, tested good on charter Latin |
| B | Court/secretary cursive 1500–1900 (wills, registers, parish) | **McCATMuS** | ✅ downloaded, tested 80–90% on clean pages |
| C | Insular minuscule, OE originals pre-1100 | **OE model** | ❌ **does not exist anywhere — the one build task** |
| D | Anything else (print, typewritten, other languages) | swap any `.mlmodel` via `--model` (TRIDIS, OICEN, print OCR…) | ⚙️ catalogue approach |

All lanes share the same engine invocation; a lane is *only* a model file +
preprocessing recipe. Deploying a new model changes nothing in the pipeline.

## Stage 3 — Preprocessing (fixed recipe, from the AGENTS.md lessons)

1. Line height target **50–120 px** → ×2–3 Lanczos upscale (tested: ×3
   sufficient, ×5 adds nothing).
2. **~20% white padding** around the page (fixes BLLA row-merging).
3. `--red-filter` only when red is decoration/cancellation, not content.
4. Never crop mid-line; flag edge-clipped lines instead.
5. Whole pages over tight crops; crops only for stubborn single lines
   (`--line`).

**Resolution is the biggest external lever.** On ~1000 px web previews we hit
a hard ceiling; real archive scans (ERO/BL, ≥2500 px wide) lift every lane.

## Stage 4 — Recognition & output

`pycatmus --batch` with the lane's `--model`, GPU. Deliverables per page:
raw `.txt` (graphematic, NFD), `--viz` confidence overlay, and — per standing
rules — raw machine output always shipped alongside any corrected text, names
never "corrected" in transcriptions.

## Stage 5 — Correction (human-in-the-loop)

pycatmus-app: hover-to-locate word boxes, amber/red underlines below 0.85/0.6
confidence, inline editing preserving lineation. This step is *also* ground
truth production — every corrected line is training data, which is why the
correction conventions below matter as much as the corrections themselves.

## Stage 6 — Training loop (ketos)

Conventions (non-negotiable, they define model quality):
- **Graphematic**: type what is inked; `ꝑ` stays `ꝑ`; no expansion, no u/v
  normalisation. CATMuS guidelines.
- **NFD** normalisation to match base models.
- Hold **2–3 pages back** from every training batch; measure CER on them.

Procedure:
```
# 1. drafts on 15–25 pages (track A) — or align ASChart/Langscape text to IIIF
#    images (track B)
pycatmus --batch "scans/*.png" --model McCATMuS.mlmodel

# 2. correct in pycatmus-app → export PageXML + images

# 3. fine-tune (base model differs per track!)
ketos train -d cuda:0 --lag 5 --normalization NFD --resize union \
  -i <base.mlmodel> -o <lane>-v1.mlmodel pages/*.xml

# 4. evaluate on held-out pages (CERberus), compare vs base
# 5. deploy: pycatmus ... --model <lane>-v1.mlmodel ; iterate
```

Expected effort/return:
- **Track A (wills/McCATMuS):** 10 corrected pages ≈ big CER jump; 30–50 pages
  ≈ corpus-grade probate model. Base model already 80–90% → fine-tune is
  optimisation, not rescue.
- **Track B (OE/insular):** insular glyphs are *new characters* to CATMuS —
  budget 25+ pages minimum; ASChart's published transcriptions of all pre-900
  charters + British Library IIIF images are the seed corpus, so the cost is
  *alignment*, not typing from scratch.

**Two models, never one merged model:** ~500 years and zero glyph overlap
between the script families; merged training dilutes both.

## Current live status (verified on this machine)

- McCATMuS fetched via `kraken get 10.5281/zenodo.13788176` (direct Zenodo
  HTTP is IP-blocked from this network) → `%LOCALAPPDATA%\pycatmus\models\McCATMuS.mlmodel`.
- Charter page (Latin + OE bounds): lane A gave usable Latin, hallucinated the
  OE — the routing split is empirically justified.
- Three will pages ×3 upscale + pad: doc 2 ~85–90% readable; docs 1 & 3 limited
  by preview resolution and secretary-hand domain → Track A fine-tune target.
- ×5 upscale vs ×3: no material gain. Recipe locked at ×3.

## Next actions (pick order)

1. Source one full-resolution will scan (ERO catalogue) → re-baseline Track A.
2. Build the manifest/export/train scripts (`scripts/` in this repo) so the
   loop is one command after corrections are done.
3. Prototype Track B corpus: pull ASChart TEI for 5 charters + matching BL
   IIIF folios, measure how much line-alignment work it really is.
4. First training runs when each track has its first 10 pages.
