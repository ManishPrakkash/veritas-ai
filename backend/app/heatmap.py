"""
Grad-CAM heatmap generation. Adapted from the validated prototype in
model/gradcam_prototype.py — see model/README.md for the experiments that
justified this approach before it was wired into the real backend.

Not production-grade: this runs an existing open SDXL-specific detector
(Organika/sdxl-detector), not our own model fine-tuned across the full
generator taxonomy in js/main.js:Classes. It only reliably localizes
SDXL-style generations (see model/README.md findings) — good enough to ship
as an opt-in "experimental" overlay, not to rely on for other generators yet.

The model is loaded once (module-level singleton, lazily on first use) and
reused across requests — reloading it per-request would be far too slow.
"""

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "Organika/sdxl-detector"
_TARGET_KEYWORDS = ("fake", "artificial", "ai", "generated", "synthetic")

# Apple system red — same accent as the extension badge's "AI-made" state
# (extension/content.js ICONS.red), so the overlay and badge read as one
# visual language rather than two different color systems.
_ACCENT_RGB = np.array([255, 59, 48], dtype=np.float32) / 255.0
_HIGHLIGHT_PERCENTILE = 75  # only the top ~25% of activation gets highlighted
_MAX_ALPHA = 0.65  # how strong the accent gets inside the highlighted region
_DIM_FACTOR = 0.88  # subtle dim on the untouched rest of the image (spotlight effect)
_EDGE_BLUR_SIGMA = 6  # soft falloff instead of a hard-edged blob

_device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = None
_model = None
_cam = None
_target_class_idx = None


class _LogitsOnlyWrapper(torch.nn.Module):
	"""pytorch_grad_cam expects model(x) to return a plain logits tensor;
	HF image-classification models return a ModelOutput object instead,
	which silently breaks GradCAM's internal handling otherwise."""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, pixel_values):
		return self.model(pixel_values=pixel_values).logits


def _auto_reshape_transform(tensor):
	"""Turn a transformer's (batch, tokens, channels) output back into a 2D
	grid, without hardcoding patch/window math for a specific architecture
	(plain ViT has a [CLS] token, Swin doesn't, etc)."""
	num_tokens = tensor.size(1)
	side = int(round(num_tokens**0.5))

	if side * side == num_tokens:
		grid = tensor
	elif side * side == num_tokens - 1:
		grid = tensor[:, 1:, :]
	else:
		raise RuntimeError(f"Can't infer a square grid from {num_tokens} tokens")

	result = grid.reshape(grid.size(0), side, side, grid.size(2))
	return result.transpose(2, 3).transpose(1, 2)


def _pick_target_layer(model):
	model_type = model.config.model_type
	if model_type in ("vit", "deit"):
		return model.vit.encoder.layer[-1].layernorm_before
	if model_type == "swin":
		return model.swin.layernorm
	raise RuntimeError(f"No known Grad-CAM target layer for model_type={model_type!r}")


def _ensure_loaded():
	global _processor, _model, _cam, _target_class_idx

	if _model is not None:
		return

	_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
	_model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
	_model.to(_device).eval()

	for idx, label in _model.config.id2label.items():
		if any(k in label.lower() for k in _TARGET_KEYWORDS):
			_target_class_idx = idx
			break
	if _target_class_idx is None:
		_target_class_idx = 0

	wrapped_model = _LogitsOnlyWrapper(_model)
	target_layers = [_pick_target_layer(_model)]
	_cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=_auto_reshape_transform)


def _render_highlight(rgb_img: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
	"""Instead of a rainbow "jet" colormap wash over the whole image, only
	highlight the specific region(s) Grad-CAM actually flagged: threshold to
	the top slice of activation, soften the edges, and tint just that area
	with a single accent color over a slightly dimmed base image (a spotlight
	effect, not a heatmap-everywhere effect)."""
	cam = grayscale_cam.astype(np.float32)
	cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

	# Binary threshold (not a gradient ramp) — the whole flagged region gets
	# full strength, so it reads as one clear highlighted area. Only the
	# blur afterward softens the boundary into a glow.
	threshold = np.percentile(cam, _HIGHLIGHT_PERCENTILE)
	mask = (cam >= threshold).astype(np.float32)
	mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=_EDGE_BLUR_SIGMA)
	mask = np.clip(mask, 0.0, 1.0)

	dimmed = rgb_img * _DIM_FACTOR
	accent_layer = np.ones_like(rgb_img) * _ACCENT_RGB
	alpha = (mask * _MAX_ALPHA)[..., None]

	result = dimmed * (1 - alpha) + accent_layer * alpha
	return np.clip(result * 255, 0, 255).astype(np.uint8)


def generate_heatmap_jpeg(image_bytes: bytes) -> bytes:
	"""Returns a JPEG-encoded heatmap overlay at the model's native input
	resolution (square). The caller stretches it back over the original
	image's on-page box — see extension/content.js."""
	_ensure_loaded()

	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	image_size = _model.config.image_size

	inputs = _processor(images=image, return_tensors="pt")
	inputs = {k: v.to(_device) for k, v in inputs.items()}

	targets = [ClassifierOutputTarget(_target_class_idx)]
	grayscale_cam = _cam(input_tensor=inputs["pixel_values"], targets=targets)[0]

	rgb_img = np.array(image.resize((image_size, image_size))).astype(np.float32) / 255.0
	visualization = _render_highlight(rgb_img, grayscale_cam)

	buffer = io.BytesIO()
	Image.fromarray(visualization).save(buffer, format="JPEG", quality=85)
	return buffer.getvalue()


def generate_heatmap_base64(image_bytes: bytes) -> str:
	return base64.b64encode(generate_heatmap_jpeg(image_bytes)).decode("ascii")
