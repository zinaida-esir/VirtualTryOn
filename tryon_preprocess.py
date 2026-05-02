# coding: utf-8
"""
Preprocessing: two raw PIL images → CP-VTON+ tensors ready for TryOnModel.

Pipeline:
  1. Fit both images to 192×256 canvas.
  2. Extract person binary mask (GrabCut or heuristic).
  3. Estimate 18-point OpenPose keypoints (KeypointRCNN).
  4. Parse person into LIP labels (SCHP if checkpoint present, else heuristic).
  5. Build shape / head (im_h) / pose_map / agnostic tensors
     exactly mirroring cp_dataset.py logic.
  6. Extract cloth mask (background subtraction + morphology).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from tryon_parse_hf import parse_human_packaged
from tryon_pose import estimate_pose

FINE_WIDTH  = 192
FINE_HEIGHT = 256

_RGB_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
_MASK_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

# LIP label groups (mirrors cp_dataset.py TOM stage)
_LBL_HEAD_GMM = {1, 4, 13}            # hat, sunglasses, face
_LBL_HEAD_TOM = {1, 2, 4, 9, 12, 13, 16, 17}  # adds hair, pants, skirt, legs
_LBL_CLOTHES  = {5, 6, 7}             # upper-clothes, dress, coat
_LBL_PERSON   = set(range(1, 20))


@dataclass
class PreparedTryOnInput:
    image:               torch.Tensor   # (3, H, W) in [-1,1]
    agnostic:            torch.Tensor   # (22, H, W)
    cloth:               torch.Tensor   # (3, H, W) in [-1,1]
    cloth_mask:          torch.Tensor   # (1, H, W) in [0,1]
    shape:               torch.Tensor   # (1, H, W) in [-1,1]
    head:                torch.Tensor   # (3, H, W) in [-1,1]
    parse_cloth:         torch.Tensor   # (3, H, W) in [-1,1]
    parse_cloth_mask:    torch.Tensor   # (1, H, W) in [0,1]
    pose_image:          torch.Tensor   # (1, H, W) in [-1,1]
    person_preview:      Image.Image
    cloth_preview:       Image.Image
    cloth_mask_preview:  Image.Image
    person_mask_preview: Image.Image
    parse_labels:        np.ndarray


def prepare_tryon_inputs(
    person_path: str | Path,
    cloth_path:  str | Path,
    # *,
    device:      str | None = None,
) -> PreparedTryOnInput:
    """
    Build all CP-VTON+ inputs from two arbitrary-resolution images.

    Parameters
    ----------
    person_path : path to full-body person photo
    cloth_path  : path to flat-lay / product garment photo
    device      : 'cpu' / 'cuda' / None (auto)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    person = _load_rgb(person_path)
    cloth  = _load_rgb(cloth_path)

    person_256 = _fit_to_canvas(person, FINE_WIDTH, FINE_HEIGHT)
    cloth_256  = _fit_to_canvas(cloth,  FINE_WIDTH, FINE_HEIGHT)

    # ── Step 1: pose (18 OpenPose keypoints) ───────────────────────────
    pose_pts = estimate_pose(person_256, device=device)          # (18, 3) float

    # ── Step 2: LIP parsing ──────────────────────────────────────────────
    # Parse the exact same 192x256 image that becomes sample.image, so parsing
    # labels, image pixels, head, parse_cloth and VTON tensors share coordinates.
    parse_array = parse_human_packaged(
        person_256,
        device=device,
        target_size=(FINE_WIDTH, FINE_HEIGHT),
    )  # (256, 192) uint8, values 0-19

    person_mask_np = _labels_to_mask(parse_array, _LBL_PERSON)
    person_mask_np = (person_mask_np * 255).astype(np.uint8)
    person_mask_np = _clean_mask(person_mask_np)

    # ── Step 3: CP-VTON+ standard tensors ───────────────────────────────
    image_t  = _RGB_TRANSFORM(person_256)
    cloth_t  = _RGB_TRANSFORM(cloth_256)

    cloth_mask_np = _cloth_mask(cloth_256)
    cloth_mask_t  = torch.from_numpy(
        (cloth_mask_np >= 128).astype(np.float32)).unsqueeze(0)

    # shape: body silhouette, down-sampled and back (like cp_dataset.py)
    shape_t      = _shape_tensor(person_mask_np)

    # head tensor (TOM mode): keep reserved labels, black out everything else
    head_t       = _head_tensor(image_t, parse_array, stage="TOM")

    # parse_cloth_mask: labels 5, 6, 7 (upper-clothes, dress, coat)
    parse_cloth_mask_np = _labels_to_mask(parse_array, _LBL_CLOTHES)
    parse_cloth_mask_t  = torch.from_numpy(
        parse_cloth_mask_np.astype(np.float32)).unsqueeze(0)

    # parse_cloth image: person image with only clothes region visible
    parse_cloth_t = image_t * parse_cloth_mask_t + (1.0 - parse_cloth_mask_t)

    # pose map: 18 × H × W heatmaps
    pose_map_t, pose_image_t = _pose_tensor(pose_pts, radius=5)

    # agnostic = [shape(1) | head(3) | pose_map(18)]  →  22 channels
    agnostic = torch.cat([shape_t, head_t, pose_map_t], dim=0)

    return PreparedTryOnInput(
        image            = image_t,
        agnostic         = agnostic,
        cloth            = cloth_t,
        cloth_mask       = cloth_mask_t,
        shape            = shape_t,
        head             = head_t,
        parse_cloth      = parse_cloth_t,
        parse_cloth_mask = parse_cloth_mask_t,
        pose_image       = pose_image_t,
        person_preview   = person_256,
        cloth_preview    = cloth_256,
        cloth_mask_preview  = Image.fromarray(cloth_mask_np, "L"),
        person_mask_preview = Image.fromarray(person_mask_np, "L"),
        parse_labels        = parse_array,
    )


# ─── Tensor builders (mirror cp_dataset.py) ──────────────────────────────────

def _shape_tensor(mask: np.ndarray) -> torch.Tensor:
    """Body silhouette: low-pass (downsample→upsample) like cp_dataset.py."""
    pil = Image.fromarray(mask, "L")
    pil = pil.resize((FINE_WIDTH // 16, FINE_HEIGHT // 16), Image.Resampling.BILINEAR)
    pil = pil.resize((FINE_WIDTH,       FINE_HEIGHT),       Image.Resampling.BILINEAR)
    return _MASK_TRANSFORM(pil)  # [-1, 1]


def _head_tensor(image_t: torch.Tensor,
                 parse_array: np.ndarray,
                 stage: str = "TOM") -> torch.Tensor:
    """
    im_h in cp_dataset.py:  im * phead  -  (1 - phead)
    Reserved labels:
      GMM: {hat, sunglasses, face}
      TOM: {hat, hair, sunglasses, pants, skirt, legs, face}
    """
    labels = _LBL_HEAD_TOM if stage == "TOM" else _LBL_HEAD_GMM
    phead = _labels_to_mask(parse_array, labels).astype(np.float32)
    p = torch.from_numpy(phead).unsqueeze(0)   # (1, H, W)  in {0, 1}
    return image_t * p - (1.0 - p)            # [-1, 1]


def _pose_tensor(points: np.ndarray,
                 radius: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 18 × H × W heatmap stack + single composite pose image."""
    pose_map   = torch.zeros((18, FINE_HEIGHT, FINE_WIDTH), dtype=torch.float32)
    pose_image = Image.new("L", (FINE_WIDTH, FINE_HEIGHT))
    pose_draw  = ImageDraw.Draw(pose_image)

    for i in range(18):
        x, y, conf = points[i]
        one  = Image.new("L", (FINE_WIDTH, FINE_HEIGHT))
        draw = ImageDraw.Draw(one)
        if conf > 0 and x > 1 and y > 1:
            box = (x - radius, y - radius, x + radius, y + radius)
            draw.rectangle(box, "white", "white")
            pose_draw.rectangle(box, "white", "white")
        pose_map[i] = _MASK_TRANSFORM(one)[0]

    return pose_map, _MASK_TRANSFORM(pose_image)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _labels_to_mask(parse_array: np.ndarray,
                    label_set: set[int]) -> np.ndarray:
    """Binary mask (0/1 float32) where parse_array ∈ label_set."""
    out = np.zeros_like(parse_array, dtype=np.float32)
    for lbl in label_set:
        out[parse_array == lbl] = 1.0
    return out


def _load_rgb(path: str | Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    return im.convert("RGB")


def _fit_to_canvas(im: Image.Image,
                   width: int, height: int,
                   fill: tuple = (255, 255, 255)) -> Image.Image:
    src = im.copy()
    src.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    x = (width  - src.width)  // 2
    y = (height - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8), iterations=2)
    # keep largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if num <= 1:
        return mask
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    best = max(areas, key=lambda t: t[1])[0]
    return (labels == best).astype(np.uint8) * 255


def _cloth_mask(im: Image.Image) -> np.ndarray:
    """Binary mask of garment on (usually white/uniform) background."""
    rgb = np.array(im)
    # estimate background colour from image border
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((dist > 24) | ((sat > 28) & (val < 248))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=1)
    # keep largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if num <= 1:
        return mask
    best = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == best).astype(np.uint8) * 255


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    x = tensor.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy(), "RGB")
