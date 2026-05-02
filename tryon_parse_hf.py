# coding: utf-8
"""Packaged FP32 SCHP/LIP parser for debugging CP-VTON+ inputs.

This loads the Hugging Face `pirocheto/schp-lip-20` model from a local
repo copy in `models/schp_hf`. It intentionally avoids the pip `schp`
package import path because that package attempts to build a CUDA extension
on import in this environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "schp_hf"

LIP_LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Glove",
    4: "Sunglasses",
    5: "Upper-clothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Jumpsuits",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe",
}

PHEAD_TOM_LABELS = {1, 2, 4, 9, 12, 13, 16, 17}
PARSE_CLOTH_LABELS = {5, 6, 7}


def parse_human_packaged(
    image: Image.Image,
    *,
    target_size: tuple[int, int] | None = None,
    device: str | None = None,
    use_fusion_logits: bool = True,
) -> np.ndarray:
    """Return a LIP label map as uint8.

    `target_size` is `(width, height)`. If omitted, the label map is resized
    back to the input image size.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = _load_model(device)
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        output = model(pixel_values=pixel_values)

    logits = output.logits if use_fusion_logits else output.parsing_logits
    out_size = target_size or image.size
    logits = F.interpolate(
        logits,
        size=(out_size[1], out_size[0]),
        mode="bilinear",
        align_corners=True,
    )
    labels = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return labels


def labels_to_mask(labels: np.ndarray, keep: set[int]) -> np.ndarray:
    return np.isin(labels, list(keep)).astype(np.uint8) * 255


@lru_cache(maxsize=1)
def _load_model(device: str):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing packaged SCHP model directory: {MODEL_DIR}")
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))

    from configuration_schp import SCHPConfig
    from image_processing_schp import SCHPImageProcessor
    from modeling_schp import SCHPForSemanticSegmentation

    config = SCHPConfig.from_pretrained(str(MODEL_DIR))
    model = SCHPForSemanticSegmentation(config)
    state = load_file(str(MODEL_DIR / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"SCHP state mismatch. Missing={missing[:5]} Unexpected={unexpected[:5]}")
    model.to(device).eval()
    processor = SCHPImageProcessor.from_pretrained(str(MODEL_DIR))
    return model, processor
