# coding: utf-8
"""FASHN human parser adapter.

The FASHN parser emits its own 18-class fashion label space. CP-VTON+ expects
LIP-style labels, so this module maps FASHN labels into the LIP ids used by
`cp_dataset.py` plus optional id 20 for skin/neck/chest.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image


FASHN_TO_LIP = {
    0: 0,   # background
    1: 13,  # face
    2: 2,   # hair
    3: 5,   # top -> upper-clothes
    4: 6,   # dress
    5: 12,  # skirt
    6: 9,   # pants
    7: 9,   # belt -> lower-body preserved-ish label
    8: 0,   # bag, no LIP equivalent useful for this pipeline
    9: 1,   # hat
    10: 11, # scarf
    11: 4,  # glasses
    12: 14, # arms, no left/right split
    13: 14, # hands -> arm/skin region
    14: 16, # legs, no left/right split
    15: 18, # feet/shoes
    16: 20, # torso/visible body -> CP-VTON+ skin/neck/chest extension
    17: 0,  # jewelry
}


def parse_human_fashn(
    image: Image.Image,
    *,
    target_size: tuple[int, int] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Return LIP-like labels as uint8."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = _load_parser(device)
    image = image.convert("RGB")
    labels = parser.predict(image)
    labels = np.asarray(labels, dtype=np.uint8)

    mapped = np.zeros_like(labels, dtype=np.uint8)
    for fashn_id, lip_id in FASHN_TO_LIP.items():
        mapped[labels == fashn_id] = lip_id

    if target_size is not None and (mapped.shape[1], mapped.shape[0]) != target_size:
        mapped = np.array(
            Image.fromarray(mapped, "L").resize(target_size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return mapped


@lru_cache(maxsize=1)
def _load_parser(device: str):
    from fashn_human_parser import FashnHumanParser

    return FashnHumanParser(device=device)
