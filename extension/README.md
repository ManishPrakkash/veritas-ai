# Veritas AI — Browser Extension (V1 scaffold)

Manifest V3 extension. Injects a small badge on eligible images as you browse; click a badge to
analyze that image. See `docs/production-roadmap.md` (repo root) for the full plan this
implements.

## Current status

This is the **client side only**. The badge's `loading` → `error` path will fire until a backend
is running at the URL configured in `background.js` (`CONFIG.backendUrl`, currently
`http://localhost:8000/api/analyze`). The backend is the next piece to build — it should:

- accept `POST` with `media` (image file) + `source_url` fields,
- call the Hive API server-side (key never lives in the extension),
- return `{ "score": 0.0-1.0, "top_class": "midjourney" }`.

The extension never calls Hive directly — see `docs/production-roadmap.md` §3 for why.

## Load it in Chrome for testing

1. Go to `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. **Load unpacked** → select this `extension/` folder
4. Visit any page with images ≥ 64×64px — a small grey badge (`V`) appears at the bottom-right
   corner of each one once it scrolls into view
5. Click a badge → it spins → turns green/yellow/red (score bands match `js/main.js:fillResults`
   in the existing site) or dark grey with `!` if the backend isn't reachable
6. **Right-click a badge** → a small settings panel opens with a "Heatmap overlay" toggle
   (persists via `chrome.storage.local`). Turn it on, then click a badge — after the normal
   green/yellow/red result, a heatmap image fades in stretched exactly over the source image,
   showing which regions the model thinks look AI-generated.

## Notes / known v1 limitations

- **Heatmap is real but experimental** — it calls `backend/app/heatmap.py`, which runs Grad-CAM
  on an existing open detector (`Organika/sdxl-detector`), not our own model trained across the
  full generator list. Per `model/README.md`'s findings: it only reliably localizes **SDXL-style**
  AI content — other generators (Midjourney, DALL-E, etc.) may show no heat at all, or heat in the
  wrong place. Don't trust it as ground truth yet; it's a working prototype, not the final model
  (roadmap §7–8, Phase 3 vs. Phase 4).
- The heatmap image is stretched to the on-page image's exact box, which distorts it slightly if
  the image's aspect ratio isn't square (the model's native input is square) — acceptable for a
  prototype, worth fixing before this ships for real.
- No auto-scan mode — analysis is manual-click only, to control backend/Hive/compute cost.
- Per-session in-memory cache only (`background.js`); real caching belongs server-side.
- `<all_urls>` host permission is required so the background service worker can fetch
  cross-origin image bytes without being subject to the page's CORS policy.
