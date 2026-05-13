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
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from tryon_parse_hf import parse_human_packaged
from tryon_parse_fashn import parse_human_fashn
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
_LBL_HEAD_TOM = {1, 2, 4, 9, 12, 13, 16, 17, 20}  # adds hair, pants, skirt, legs, optional neck/skin
_LBL_CLOTHES  = {5, 6, 7}             # upper-clothes, dress, coat
_LBL_PERSON   = set(range(1, 21))


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
    neck_mask:           torch.Tensor   # (1, H, W) in [0,1]
    person_preview:      Image.Image
    cloth_preview:       Image.Image
    cloth_mask_preview:  Image.Image
    person_mask_preview: Image.Image
    neck_mask_preview:   Image.Image
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
    parse_array = _parse_human(
        person_256,
        device=device,
        target_size=(FINE_WIDTH, FINE_HEIGHT),
    )  # (256, 192) uint8, LIP-like values

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

    # head tensor (TOM mode): keep reserved labels and a synthetic neck patch,
    # otherwise TOM tends to hallucinate a pale collar/neck boundary.
    neck_mask_np = _synthetic_neck_mask(parse_array, pose_pts, person_256)
    head_t       = _head_tensor(image_t, parse_array, stage="TOM", extra_mask=neck_mask_np)

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
        neck_mask        = torch.from_numpy(neck_mask_np.astype(np.float32)).unsqueeze(0),
        person_preview   = person_256,
        cloth_preview    = cloth_256,
        cloth_mask_preview  = Image.fromarray(cloth_mask_np, "L"),
        person_mask_preview = Image.fromarray(person_mask_np, "L"),
        neck_mask_preview   = Image.fromarray((neck_mask_np * 255).astype(np.uint8), "L"),
        parse_labels        = parse_array,
    )


# ─── Tensor builders (mirror cp_dataset.py) ──────────────────────────────────

def _shape_tensor(mask: np.ndarray) -> torch.Tensor:
    """Body silhouette: low-pass (downsample→upsample) like cp_dataset.py."""
    pil = Image.fromarray(mask, "L")
    pil = pil.resize((FINE_WIDTH // 16, FINE_HEIGHT // 16), Image.Resampling.BILINEAR)
    pil = pil.resize((FINE_WIDTH,       FINE_HEIGHT),       Image.Resampling.BILINEAR)
    return _MASK_TRANSFORM(pil)  # [-1, 1]


def _parse_human(image: Image.Image,
                 *,
                 device: str | None,
                 target_size: tuple[int, int]) -> np.ndarray:
    parser = os.environ.get("VTON_HUMAN_PARSER", "fashn").strip().lower()
    if parser == "schp":
        return parse_human_packaged(image, device=device, target_size=target_size)
    if parser != "fashn":
        raise ValueError("VTON_HUMAN_PARSER must be 'fashn' or 'schp'")
    return parse_human_fashn(image, device=device, target_size=target_size)


def _head_tensor(image_t: torch.Tensor,
                 parse_array: np.ndarray,
                 stage: str = "TOM",
                 extra_mask: np.ndarray | None = None) -> torch.Tensor:
    """
    im_h in cp_dataset.py:  im * phead  -  (1 - phead)
    Reserved labels:
      GMM: {hat, sunglasses, face}
      TOM: {hat, hair, sunglasses, pants, skirt, legs, face}
    """
    labels = _LBL_HEAD_TOM if stage == "TOM" else _LBL_HEAD_GMM
    phead = _labels_to_mask(parse_array, labels).astype(np.float32)
    if extra_mask is not None:
        phead = np.maximum(phead, extra_mask.astype(np.float32))
    p = torch.from_numpy(phead).unsqueeze(0)   # (1, H, W)  in {0, 1}
    return image_t * p - (1.0 - p)            # [-1, 1]


def _synthetic_neck_mask(parse_array: np.ndarray,
                         pose_points: np.ndarray | None,
                         image: Image.Image) -> np.ndarray:
    """Preserve the original neck/chin skin region for TOM.

    LIP parsers often do not expose CP-VTON+'s added label 20
    (skin/neck/chest). We synthesize a conservative neck patch from the face
    label and OpenPose neck point so the renderer cannot repaint it.
    """
    face = parse_array == 13
    ys, xs = np.where(face)
    mask = np.zeros_like(parse_array, dtype=np.float32)
    if len(xs) == 0:
        return mask

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    center_x = int((x1 + x2) / 2)

    neck_y = y2 + int(0.35 * face_h)
    if pose_points is not None and pose_points.shape[0] > 1 and pose_points[1, 2] > 0:
        neck_y = int(pose_points[1, 1])

    top_y = max(0, y2 - int(0.08 * face_h))
    bottom_y = min(parse_array.shape[0] - 1, max(neck_y + int(0.18 * face_h), y2 + int(0.35 * face_h)))
    top_half = max(2, int(0.22 * face_w))
    bottom_half = max(top_half + 1, int(0.38 * face_w))

    polygon = np.array([
        [center_x - top_half, top_y],
        [center_x + top_half, top_y],
        [center_x + bottom_half, bottom_y],
        [center_x - bottom_half, bottom_y],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1.0)
    mask = mask * _skin_color_mask(image, face, mask > 0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return np.clip(mask, 0.0, 1.0)


def _skin_color_mask(image: Image.Image,
                     face_mask: np.ndarray,
                     candidate_mask: np.ndarray) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    if face_mask.sum() < 8:
        return candidate_mask.astype(np.float32)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    face_pixels = ycrcb[face_mask]
    ref = np.median(face_pixels[:, 1:3], axis=0)
    chroma_dist = np.linalg.norm(ycrcb[:, :, 1:3] - ref[None, None, :], axis=2)
    luminance = ycrcb[:, :, 0]
    skin = (chroma_dist < 24) & (luminance > 45) & candidate_mask
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return skin.astype(np.float32)


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
