import hashlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .heatmap import generate_heatmap_base64
from .hive_client import HiveError, analyze as hive_analyze

app = FastAPI(title="Veritas AI backend")

# The extension's background service worker is the only v1 caller. Loosened
# for local dev; tighten to the extension's origin before this goes anywhere
# public.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["POST", "GET"],
	allow_headers=["*"],
)

# v1: in-memory, per-process cache keyed by image content hash. Good enough
# for local dev; swap for Redis (or similar) before running more than one
# worker process, since this dict isn't shared across processes.
_result_cache: dict[str, dict] = {}
_heatmap_cache: dict[str, str] = {}


@app.get("/healthz")
async def healthz():
	return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(media: UploadFile = File(...), source_url: str = Form(None)):
	image_bytes = await media.read()

	if not image_bytes:
		raise HTTPException(status_code=400, detail="Empty upload")

	image_hash = hashlib.sha256(image_bytes).hexdigest()

	if image_hash in _result_cache:
		return _result_cache[image_hash]

	try:
		top_class, score = await hive_analyze(image_bytes, media.filename or "image.jpg")
	except HiveError as error:
		raise HTTPException(status_code=502, detail=str(error))

	result = {"score": score, "top_class": top_class}
	_result_cache[image_hash] = result
	return result


@app.post("/api/heatmap")
async def heatmap(media: UploadFile = File(...)):
	"""Experimental — see app/heatmap.py docstring. Only reliably localizes
	SDXL-style AI generation right now, not the full generator taxonomy."""
	image_bytes = await media.read()

	if not image_bytes:
		raise HTTPException(status_code=400, detail="Empty upload")

	image_hash = hashlib.sha256(image_bytes).hexdigest()

	if image_hash in _heatmap_cache:
		return {"heatmap_base64": _heatmap_cache[image_hash]}

	try:
		heatmap_b64 = generate_heatmap_base64(image_bytes)
	except Exception as error:  # noqa: BLE001 - surface as a clean 500 either way
		raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {error}")

	_heatmap_cache[image_hash] = heatmap_b64
	return {"heatmap_base64": heatmap_b64}
