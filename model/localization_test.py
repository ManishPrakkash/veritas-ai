"""
Phase 3 localization sanity check: build a synthetic "partially AI" test
image (a real photo with a known rectangular region replaced by genuine
AI-generated content), run the existing Grad-CAM prototype's classifier on
it, and check whether the heatmap's "hot" pixels actually concentrate inside
the known AI region rather than spreading evenly across the real background.

This deliberately does NOT use a real inpainting model (no diffusion model
download, no blended edit) — compositing a hard-edged patch is a cheap proxy
good enough to sanity-check whether classifier-level Grad-CAM carries any
real localization signal at all, before investing in the much bigger effort
of a proper segmentation model trained on true inpainting masks (see
docs/production-roadmap.md, Phase 4).

Usage:
    python model/localization_test.py --real <path_or_url> --ai <path_or_url> [--box x0,y0,x1,y1]

--box is fractional (0-1) coordinates of the pasted region; default is the
bottom-right quarter of the image.
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.dirname(__file__))
from gradcam_prototype import (  # noqa: E402
	LogitsOnlyWrapper,
	SAMPLES_DIR,
	auto_reshape_transform,
	load_model,
	pick_target_layer,
)


def load_image(source):
	os.makedirs(SAMPLES_DIR, exist_ok=True)

	if source.startswith("http://") or source.startswith("https://"):
		tmp_path = os.path.join(SAMPLES_DIR, "_fetched_" + os.path.basename(source.split("?")[0]))
		urllib.request.urlretrieve(source, tmp_path)
		return Image.open(tmp_path).convert("RGB")

	return Image.open(source).convert("RGB")


def build_composite(real_img, ai_img, box_fractional, size):
	composite = real_img.resize((size, size))

	x0 = int(box_fractional[0] * size)
	y0 = int(box_fractional[1] * size)
	x1 = int(box_fractional[2] * size)
	y1 = int(box_fractional[3] * size)

	patch = ai_img.resize((x1 - x0, y1 - y0))
	composite.paste(patch, (x0, y0))

	mask = np.zeros((size, size), dtype=bool)
	mask[y0:y1, x0:x1] = True

	return composite, mask, (x0, y0, x1, y1)


def find_target_class_index(id2label, probs):
	for idx, label in id2label.items():
		if any(k in label.lower() for k in ("fake", "artificial", "ai", "generated", "synthetic")):
			return idx
	return int(probs.argmax().item())


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--real", required=True, help="Path or URL to a real photo")
	parser.add_argument("--ai", required=True, help="Path or URL to a genuine AI-generated image")
	parser.add_argument("--box", default="0.5,0.5,1.0,1.0", help="x0,y0,x1,y1 as fractions (default: bottom-right quarter)")
	args = parser.parse_args()

	box = tuple(float(v) for v in args.box.split(","))

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_id, processor, model = load_model()
	model.to(device).eval()
	print("Loaded:", model_id, "on", device)

	image_size = model.config.image_size

	real_img = load_image(args.real)
	ai_img = load_image(args.ai)
	composite, mask, box_px = build_composite(real_img, ai_img, box, image_size)

	composite_path = os.path.join(SAMPLES_DIR, "composite.jpg")
	composite.save(composite_path)

	inputs = processor(images=composite, return_tensors="pt")
	inputs = {k: v.to(device) for k, v in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits
		probs = torch.softmax(logits, dim=-1)[0]

	print("Composite predictions:")
	for idx, label in model.config.id2label.items():
		print(f"  {label}: {probs[idx].item() * 100:.2f}%")

	target_idx = find_target_class_index(model.config.id2label, probs)
	print(f"Grad-CAM target class: {model.config.id2label[target_idx]} (index {target_idx})")

	target_layers = [pick_target_layer(model)]
	wrapped_model = LogitsOnlyWrapper(model)
	cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=auto_reshape_transform)
	targets = [ClassifierOutputTarget(target_idx)]

	grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=targets)[0]

	mean_inside = float(grayscale_cam[mask].mean())
	mean_outside = float(grayscale_cam[~mask].mean())
	ratio = mean_inside / (mean_outside + 1e-8)

	print()
	print(f"Mean heatmap activation INSIDE known-AI region:  {mean_inside:.4f}")
	print(f"Mean heatmap activation OUTSIDE (real background): {mean_outside:.4f}")
	print(f"Localization ratio (inside/outside, >1 = concentrating correctly): {ratio:.2f}x")

	rgb_img = np.array(composite).astype(np.float32) / 255.0
	visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

	vis_img = Image.fromarray(visualization)
	draw = ImageDraw.Draw(vis_img)
	x0, y0, x1, y1 = box_px
	draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255), width=2)

	heatmap_path = os.path.join(SAMPLES_DIR, "localization_heatmap.jpg")
	vis_img.save(heatmap_path)

	print()
	print("Saved composite to:", composite_path)
	print("Saved heatmap (white box = known-AI region) to:", heatmap_path)


if __name__ == "__main__":
	main()
