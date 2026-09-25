# Veritas AI — Extension & Detection Pipeline: Production Roadmap

## 1. Vision

Turn Veritas AI from a "paste an image into a website" tool into a **browser extension** that
scores images *in place* as you browse, and eventually shows **which pixels** of an image were
AI-generated — not just a single yes/no percentage.

Two capabilities, shipped in stages:

1. **Detection** — is this image AI-generated? (V1, ships first, powered by Hive)
2. **Localization ("heatmap")** — *where* in the image is it AI-generated? (V2+, requires our own
   fine-tuned model — Hive's API cannot do this)

## 2. V1 Scope

- Manifest V3 Chrome extension.
- Content script scans the page and injects a small badge icon on top of `<img>` elements
  (skip tiny/icon-sized images — favicons, avatars, emoji — to avoid clutter).
- Badge starts **neutral/grey** (unanalyzed). **Manual trigger**: user clicks the badge to run
  detection (no auto-scan-everything in v1 — keeps API cost and page noise down).
- On click: badge → loading spinner → **red / yellow / green** based on score.
- No heatmap in v1. That's the whole point of doing v1 first: ship the traffic-light version on
  the existing Hive API while the localization model is being built in parallel.

## 3. Why a Backend Server Now (vs. today's client-only proxy hack)

The current site (`js/main.js`) calls Hive directly from the browser through public CORS proxies
(`proxy.hovida.de`, `proxy.mein-chatserver.de`) with no API key. That's fine for a demo page, but
wrong for an extension that will run on every page a user visits:

- **API key custody** — Hive (and later, our own inference server) needs a real key/auth; that
  cannot live in extension JS, which is trivially readable by anyone.
- **Caching / cost control** — the same viral image gets requested by thousands of users; a
  server can hash-and-cache a result instead of re-billing Hive every time.
- **Swap-ability** — the extension should only ever talk to *our* API. Whether that API is backed
  by Hive today or our own model tomorrow is an implementation detail the client never needs to
  know about. This is the seam that makes the V1 → V2 migration painless.

So: **extension → our backend → Hive API → response → extension**, not extension → Hive directly.

## 4. End-to-End Flow (V1)

```
 1. Page loads, content script finds <img> tags
 2. Badge icon injected (grey) on eligible images
 3. User clicks badge
 4. Content script grabs the image (fetch the src, or canvas-draw if it's same-origin;
    cross-origin no-cors images need the background service worker to fetch)
 5. Background service worker POSTs the image to our backend: POST /api/analyze
 6. Backend:
      a. hash the image (content hash) → cache lookup → return cached verdict if hit
      b. on miss: call Hive API server-side (API key attached here, not in the extension)
      c. map Hive's per-class scores → single red/yellow/green verdict (see §5)
      d. cache result by hash, return { verdict, score, topClass }
 7. Background service worker relays response to content script
 8. Badge updates: grey → red | yellow | green (+ tooltip with score / top class)
```

## 5. Traffic-Light Thresholds

Reuse the banding already implemented in `js/main.js:fillResults` (score = highest AI-class
probability × 100), just collapsed from 3 labels into 3 colors:

| Score range | Existing label (main.js) | V1 badge color |
|---|---|---|
| < 1.5% | "Not AI Generated" | 🟢 Green |
| 1.5% – 10% | "Edited" | 🟡 Yellow |
| > 10% | "AI Generated" | 🔴 Red |

Keep these as a backend-side config (not hardcoded in the extension) so thresholds can be tuned
without shipping a new extension version.

## 6. Extension Architecture Notes (Manifest V3)

- `content_scripts`: injected on `document_idle`, uses `MutationObserver` +
  `IntersectionObserver` to catch lazy-loaded / infinite-scroll images, not just the initial DOM.
- `background` (service worker): owns all `fetch()` calls to our backend — content scripts can
  hit page CSP restrictions and cross-origin fetch limits that a service worker doesn't.
- Permissions: `activeTab`, `scripting`, `storage` (cache verdicts per-tab so re-scanning the same
  page doesn't re-hit the backend), `host_permissions` scoped to our backend's domain.
- **Naming collision to fix first**: the existing `manifest.json` at repo root is a *PWA* manifest
  (icons/theme_color for the web app), not a Chrome extension manifest. A Chrome extension
  manifest (`manifest_version: 3`, `permissions`, `background`, `content_scripts`, etc.) is a
  different schema and can't reuse that file. Put the extension in its own directory (e.g.
  `extension/`) with its own `manifest.json`, separate from the static site at the repo root.

## 7. Why Hive Alone Can't Get Us to Heatmaps

Hive's `ai_detection` endpoint returns **global, image-level class probabilities**
(`sora`, `midjourney`, `stablediffusion`, …, each 0–1) — no bounding boxes, no per-pixel/segment
output. There's no API parameter that will make it localize *where* in the image the AI content
is. Getting a heatmap requires a model where locality is part of the architecture or the
inference procedure — i.e., our own model.

## 8. Heatmap / Localization Strategy (V2+)

Three viable approaches, roughly in order of "fastest to ship" → "most accurate":

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Grad-CAM / Grad-CAM++ / Score-CAM** on a fine-tuned classifier | Backprop gradients (or perturbation scores) through a CNN classifier's last conv layer to get a coarse class-activation map | No mask-labeled data needed — works on top of a plain fine-tuned classifier we're building anyway for detection | Coarse/blurry localization, not pixel-accurate |
| **Patch/tile classifier** | Slice the image into overlapping patches, classify each patch independently, stitch scores into a heatmap grid | Simple, no new architecture, decent granularity, this is what most public "AI heatmap" tools (Illuminarty, AI-or-Not) actually do | Blocky edges, loses global context per patch, N× inference cost |
| **Pixel-wise segmentation fine-tune** | Train a segmentation head (e.g. U-Net-style decoder) to directly predict a real-vs-AI mask | Most accurate, true pixel-level output | Needs **mask-labeled training data** — the hard part (see §8.3) |

**Recommendation**: ship Grad-CAM on the V1 classifier first (near-zero extra work once the
classifier exists), then invest in the segmentation model once we have masked training data. Use
the patch approach as a fallback/sanity-check baseline in between.

### 8.1 Base Model Candidates (for the fine-tuned detector)

| Model family | Why it's a candidate |
|---|---|
| **CLIP (ViT) + linear/MLP probe** — "universal fake detector" style | CLIP features generalize well *across* generators (trained on one generator's outputs, still catches others) — matches our need to cover ~70 different generator classes and counting |
| **EfficientNet / Xception fine-tune** | Standard backbone in deepfake/AI-detection literature, cheap to fine-tune, works well with Grad-CAM |
| **Frequency-aware detectors** (e.g. CNNDetection, DIRE — diffusion reconstruction error, NPR — neighboring pixel relationships) | Diffusion/GAN models leave frequency-domain artifacts invisible to plain RGB classifiers; these architectures specifically exploit that signal and tend to generalize better to *unseen* generators |

Practical default: start from a **CLIP ViT-B backbone + fine-tuned classification head**
(generalizes best across our long generator list), add Grad-CAM for v2 heatmaps, and evaluate a
frequency-aware model (DIRE-style) in parallel since most modern generators in our class list
(Sora, Kling, Luma, etc.) are diffusion-based.

### 8.2 Data Strategy

- **Real images**: COCO / OpenImages / FFHQ (diverse, well-established negatives).
- **AI images, per class**: sample outputs from each generator already enumerated in
  `js/main.js:Classes` (Midjourney, SDXL, DALL-E, Sora, Kling, Runway, …) — the taxonomy is
  already defined in the existing codebase, reuse it as the label set.
- **Masked/localization data (the hard part)**: generate it ourselves — take real images, use
  inpainting (Stable Diffusion / SDXL inpainting) to AI-edit only *part* of the image, and keep
  the inpainting mask as ground truth. This gives free, exact pixel masks for "partially AI"
  images without manual annotation. Public partial-edit datasets (AutoSplice, CocoGlide) can
  supplement this.

### 8.3 Training Phases

1. Fine-tune classifier (base model + head) on full-image real-vs-AI-class data.
2. Add Grad-CAM on top of (1) — first heatmap, zero extra labeled data.
3. Fine-tune a segmentation head on the self-generated inpainting-mask dataset — pixel-accurate
   heatmap.
4. Continual retraining as new generators appear — extend the class list (already an ongoing
   pattern in `main.js:Classes`, which already tracks ~70 generators).

## 9. Phased Roadmap

| Phase | Deliverable |
|---|---|
| 0 (current) | Static site, client-side Hive calls through public proxies |
| **1 (next)** | MV3 extension, manual-click badge, backend service wrapping Hive (key custody + caching), red/yellow/green |
| 2 | Optional auto-scan mode, per-domain allow/block list, bulk "scan all images on page" |
| 3 | Grad-CAM heatmap from our own fine-tuned classifier (augments/replaces Hive per-image) |
| 4 | Segmentation-based pixel-accurate heatmap, trained on self-generated masked data |
| 5 | Continuous retraining pipeline as new generators launch |

## 10. Open Decisions / Risks

- **Backend stack**: **Python (FastAPI)** — confirmed. The inference server in Phase 3+ will be
  PyTorch-based anyway, so one stack avoids a rewrite at the service boundary. Scaffolded in
  `backend/`.
- **Hive access**: the existing site (and the Phase 1 backend) calls Hive's public, unauthenticated
  "plugin" endpoint — the same one Hive's own browser extension uses — not a keyed/billed
  developer API. There's no confirmed pricing or usage terms for this specific endpoint; treat it
  as provisional (see `backend/app/hive_client.py`). Hash-based caching is still worth keeping
  regardless — it reduces load on an endpoint we don't control and speeds up repeat lookups — but
  don't assume a per-call bill exists until Hive's official API is actually adopted.
- **Privacy/ToS**: the extension will send images from arbitrary pages (including ones the user
  didn't upload themselves) to our server and to Hive. Needs a clear privacy note in the
  extension listing and should respect obviously-private contexts (e.g. skip images inside
  password-protected/webmail-style pages if feasible).
- **False positives on real photos with heavy edits/filters** — the existing "Edited" middle band
  already anticipates this; keep it in the heatmap version too rather than forcing a binary
  verdict.

## 11. Immediate Next Steps

1. Scaffold `extension/` (MV3 manifest, content script, background service worker) — badge
   injection + click-to-analyze UI, calling a stubbed backend endpoint.
2. Stand up the backend `/api/analyze` endpoint wrapping the existing Hive call (server-side key,
   hash-based cache), reusing the scoring/threshold logic from `js/main.js`.
3. Wire the two together end-to-end for v1 (no heatmap yet).
4. Kick off the Grad-CAM classifier fine-tune in parallel once v1 is stable.
