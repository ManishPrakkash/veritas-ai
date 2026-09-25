# Veritas AI — Heatmap Prototype (Phase 3 groundwork)

Early experiments toward `docs/production-roadmap.md` Phase 3 (Grad-CAM heatmap) and validating
the approach before investing in our own fine-tuned model. **Nothing here ships in the
extension** — this is offline research/validation only.

## Scripts

- `gradcam_prototype.py` — loads an existing open AI-image-detection model from Hugging Face
  (currently `Organika/sdxl-detector`, a Swin Transformer; falls back to
  `umm-maybe/AI-image-detector`) and runs Grad-CAM on a single image, saving a heatmap overlay.
  Handles both plain-ViT and Swin-style backbones (auto-detects grid size from the actual token
  count rather than hardcoding patch-size math).
- `localization_test.py` — the real question: **does the heatmap actually find the AI-generated
  part of an image, or just light up randomly?** Builds a synthetic ground-truth test case (a
  real photo + a genuine AI-generated image pasted into a known region), runs the same Grad-CAM
  pipeline, and scores it: mean heatmap activation inside the known-fake region vs. outside it.
  Ratio > 1 means real localization signal; ~0 means none.

Run either with `python model/<script>.py --help`.

## Findings so far (2026-09-25)

| Test | AI patch domain | Overall verdict | Inside/outside ratio |
|---|---|---|---|
| Midjourney patch on a real photo | Mismatch (detector only knows SDXL) | Missed entirely (99.93% "human") | 0.00x — no signal |
| SDXL patch, bottom-right quadrant | Match | Correctly caught (99.18% "artificial") | **2.07x** — real, coarse localization |
| SDXL patch, top-left quadrant (different photo) | Match | Correctly caught (99.87% "artificial") | **1.44x** — heat moved with the patch, ruling out a fixed-corner positional bias |

**Takeaway**: classifier-level Grad-CAM does carry genuine spatial signal — but only for
generators the underlying classifier actually recognizes. A single narrow off-the-shelf detector
(this one only knows SDXL) is blind to everything else. This is direct evidence for the roadmap's
plan to fine-tune our own classifier across the full generator taxonomy already tracked in
`js/main.js:Classes`, rather than depending on someone else's narrow one — and confirms Grad-CAM
is worth building on top of once that classifier exists, rather than needing to jump straight to
a full segmentation model.

Also confirmed empirically (not just architecturally): the localization is coarse, with heat
bleeding into unrelated *complex/high-detail* regions of the real photo (not just uniformly
wrong) — consistent with the roadmap's expectation that this is a "roughly the right area,"
not pixel-accurate, technique.

## Environment notes

- Runs on CPU fine for single-image tests (a few seconds each). GPU (CUDA build of torch) was
  attempted but isn't required for this stage — see conversation history if revisiting; the
  machine has an RTX 2050 (4GB) if/when GPU acceleration becomes worth the larger download.
- Test images are fetched via public Hugging Face dataset-viewer API rows (verified, labeled
  samples) or picsum.photos — never fabricated/guessed URLs. `model/samples/` is gitignored;
  nothing there is committed.
