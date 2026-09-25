"""
Heatmap prototype, step 1: prove the pipeline (model -> Grad-CAM -> heatmap
image) works at all, using an existing open AI-image-detection model instead
of one we've trained ourselves.

This is NOT the model Veritas AI will ship with. Quality/accuracy from this
specific checkpoint is not the point yet — the point is validating that we
can compute a class-activation heatmap from a HuggingFace image-classification
model and turn it into an overlay image, before investing in our own
fine-tune (see docs/production-roadmap.md, Phase 3).

Run:
    python model/gradcam_prototype.py [image_path_or_url]
"""

import os
import sys
import urllib.request

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Candidates in priority order — public ViT-based "is this AI-generated"
# classifiers on the Hub. Tried in order in case one is unavailable/gated.
MODEL_CANDIDATES = [
	"Organika/sdxl-detector",
	"umm-maybe/AI-image-detector",
]

DEFAULT_SAMPLE_URL = "https://picsum.photos/512"
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def load_model():
	last_error = None
	for model_id in MODEL_CANDIDATES:
		try:
			print(f"Trying model: {model_id}")
			processor = AutoImageProcessor.from_pretrained(model_id)
			model = AutoModelForImageClassification.from_pretrained(model_id)
			return model_id, processor, model
		except Exception as error:  # noqa: BLE001 - want to try the next candidate
			print(f"  failed: {error}")
			last_error = error
	raise RuntimeError(f"Could not load any candidate model: {last_error}")


def load_sample_image(source):
	os.makedirs(SAMPLES_DIR, exist_ok=True)

	if source and (source.startswith("http://") or source.startswith("https://")):
		path = os.path.join(SAMPLES_DIR, "downloaded.jpg")
		urllib.request.urlretrieve(source, path)
	elif source:
		path = source
	else:
		path = os.path.join(SAMPLES_DIR, "downloaded.jpg")
		if not os.path.exists(path):
			urllib.request.urlretrieve(DEFAULT_SAMPLE_URL, path)

	return Image.open(path).convert("RGB")


def find_target_class_index(id2label):
	"""Pick whichever label looks like the "this is AI-generated" class."""
	keywords = ["fake", "artificial", "ai", "generated", "synthetic"]
	for idx, label in id2label.items():
		if any(k in label.lower() for k in keywords):
			return idx
	return None


def auto_reshape_transform(tensor):
	"""Turn a transformer's (batch, tokens, channels) output back into a 2D
	grid Grad-CAM can treat like a conv feature map — without needing to know
	the exact patch/window math for whatever backbone this is (plain ViT has
	a [CLS] token, Swin doesn't, etc). Just look at the actual token count."""
	num_tokens = tensor.size(1)
	side = int(round(num_tokens ** 0.5))

	if side * side == num_tokens:
		grid = tensor  # no [CLS] token — e.g. Swin
	elif side * side == num_tokens - 1:
		grid = tensor[:, 1:, :]  # one [CLS] token — e.g. plain ViT
	else:
		raise RuntimeError(f"Can't infer a square grid from {num_tokens} tokens")

	result = grid.reshape(grid.size(0), side, side, grid.size(2))
	result = result.transpose(2, 3).transpose(1, 2)
	return result


class LogitsOnlyWrapper(torch.nn.Module):
	"""pytorch_grad_cam expects model(x) to return a plain logits tensor;
	HF image-classification models return a ModelOutput object instead
	(which iterates as dict keys, not values — silently breaks GradCAM's
	internal zip() over "outputs" otherwise). Target layers are still looked
	up on the wrapped model, so hooks attach to the real modules."""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, pixel_values):
		return self.model(pixel_values=pixel_values).logits


def pick_target_layer(model):
	model_type = model.config.model_type
	if model_type in ("vit", "deit"):
		return model.vit.encoder.layer[-1].layernorm_before
	if model_type == "swin":
		return model.swin.layernorm
	raise RuntimeError(
		f"No known Grad-CAM target layer for model_type={model_type!r}. "
		"Pick a different candidate model or add one for this architecture."
	)


def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print("Device:", device, (torch.cuda.get_device_name(0) if device == "cuda" else ""))

	model_id, processor, model = load_model()
	model.to(device).eval()
	print("Loaded:", model_id)
	print("Labels:", model.config.id2label)

	image_arg = sys.argv[1] if len(sys.argv) > 1 else None
	image = load_sample_image(image_arg)

	inputs = processor(images=image, return_tensors="pt")
	inputs = {k: v.to(device) for k, v in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits
		probs = torch.softmax(logits, dim=-1)[0]

	print("Predictions:")
	for idx, label in model.config.id2label.items():
		print(f"  {label}: {probs[idx].item() * 100:.2f}%")

	target_idx = find_target_class_index(model.config.id2label)
	if target_idx is None:
		target_idx = int(probs.argmax().item())
	print(f"Grad-CAM target class: {model.config.id2label[target_idx]} (index {target_idx})")

	image_size = model.config.image_size
	target_layers = [pick_target_layer(model)]
	wrapped_model = LogitsOnlyWrapper(model)

	cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=auto_reshape_transform)
	targets = [ClassifierOutputTarget(target_idx)]

	grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=targets)[0]

	rgb_img = np.array(image.resize((image_size, image_size))).astype(np.float32) / 255.0
	visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

	output_path = os.path.join(SAMPLES_DIR, "heatmap.jpg")
	Image.fromarray(visualization).save(output_path)
	print("Saved heatmap to:", output_path)


if __name__ == "__main__":
	main()
