# coding: utf-8
"""
Human parsing with LIP-compatible labels (20 classes).

Primary:  SCHP AugmentCE2P (ResNet-101 backbone).
          Requires checkpoint at  models/schp_lip.pth
          Download: https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
          (LIP_final_Training_model.pth from the SCHP repo)

Fallback: Spatial reasoning from KeypointRCNN keypoints + person mask.

LIP label map:
  0  Background     1  Hat           2  Hair         3  Glove
  4  Sunglasses     5  UpperClothes  6  Dress        7  Coat
  8  Socks          9  Pants         10 Jumpsuits     11 Scarf
  12 Skirt          13 Face          14 LeftArm       15 RightArm
  16 LeftLeg        17 RightLeg      18 LeftShoe      19 RightShoe

Architecture reverse-engineered from actual checkpoint keys/shapes.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── Building blocks ─────────────────────────────────────────────────────────

class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 dilation: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


def _make_layer(inplanes: int, planes: int, blocks: int, stride: int = 1,
                dilation: int = 1, multi_grid: tuple = (1, 1, 1)):
    out_ch = planes * _Bottleneck.expansion
    downsample = None
    if stride != 1 or inplanes != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    layers = [_Bottleneck(inplanes, planes, stride=stride,
                          dilation=dilation * multi_grid[0],
                          downsample=downsample)]
    for i in range(1, blocks):
        mg = multi_grid[i % len(multi_grid)]
        layers.append(_Bottleneck(out_ch, planes, dilation=dilation * mg))
    return nn.Sequential(*layers), out_ch


class _PSPModule(nn.Module):
    """Pyramid Spatial Pooling.  bottleneck input = features + per*4 = 4096."""

    def __init__(self, features: int = 2048, out_features: int = 512,
                 sizes: tuple = (1, 2, 3, 6)):
        super().__init__()
        per = features // len(sizes)   # 512
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),                          # idx 0
                nn.Conv2d(features, per, 1, bias=False),          # idx 1
                nn.BatchNorm2d(per),                               # idx 2
                nn.ReLU(inplace=True),                             # idx 3
            ) for s in sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + per * len(sizes), out_features,  # idx 0
                      3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),                          # idx 1
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        priors = [F.interpolate(s(x), (h, w), mode='bilinear',
                                align_corners=True) for s in self.stages]
        return self.bottleneck(torch.cat([x] + priors, dim=1))


class _Edge(nn.Module):
    """Multi-scale edge module.  Named 'edge' in the checkpoint (not 'edge_layer')."""

    def __init__(self):
        super().__init__()
        # each conv_* is Sequential(Conv1×1, BN);  ReLU applied in forward
        self.conv1 = nn.Sequential(                             # takes l1 (256ch)
            nn.Conv2d(256,  256, 1, bias=False), nn.BatchNorm2d(256))
        self.conv2 = nn.Sequential(                             # takes l2 (512ch)
            nn.Conv2d(512,  256, 1, bias=False), nn.BatchNorm2d(256))
        self.conv3 = nn.Sequential(                             # takes l3 (1024ch)
            nn.Conv2d(1024, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.conv4 = nn.Conv2d(256, 2, 3, padding=1)           # shared edge head
        self.conv5 = nn.Conv2d(6,   2, 1)                      # fuse 3×2 → 2

    def forward(self, l1: torch.Tensor, l2: torch.Tensor,
                l3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = l3.shape
        e1 = self.conv4(F.relu(self.conv1(
            F.interpolate(l1, (h, w), mode='bilinear', align_corners=True))))
        e2 = self.conv4(F.relu(self.conv2(
            F.interpolate(l2, (h, w), mode='bilinear', align_corners=True))))
        e3 = self.conv4(F.relu(self.conv3(l3)))
        edge = self.conv5(torch.cat([e1, e2, e3], dim=1))
        return edge, e3


class _Decoder(nn.Module):
    """Parsing decoder.  conv3 uses all 1×1 convs (no padding)."""

    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.conv1 = nn.Sequential(                            # context 512→256
            nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.conv2 = nn.Sequential(                            # l1 256→48
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48))
        # 4 parameter layers → indices 0,1,2,3 in state dict (no ReLU inside)
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, 1, bias=False),                # idx 0
            nn.BatchNorm2d(256),                                # idx 1
            nn.Conv2d(256, 256, 1, bias=False),                # idx 2
            nn.BatchNorm2d(256),                                # idx 3
        )
        self.conv4 = nn.Conv2d(256, num_classes, 1)            # with bias

    def forward(self, context: torch.Tensor,
                l1: torch.Tensor) -> torch.Tensor:
        _, _, h, w = l1.shape
        xt = F.interpolate(F.relu(self.conv1(context)),
                           (h, w), mode='bilinear', align_corners=True)
        xl = F.relu(self.conv2(l1))
        x  = self.conv3(torch.cat([xt, xl], dim=1))  # conv3 has no internal ReLU
        return self.conv4(F.relu(x))


# ─── Full model ───────────────────────────────────────────────────────────────

class AugmentCE2P(nn.Module):
    """SCHP AugmentCE2P — architecture matched to LIP_final_Training_model.pth.

    Input:  (N, 3, 473, 473)  BGR normalised with MEAN/STD below.
    Output: (N, 20, 473, 473) parsing logits,  (N, 2, 473, 473) edge logits.
    Use argmax on parsing to get LIP label map.
    """

    MEAN = (0.406, 0.456, 0.485)   # BGR channel order
    STD  = (0.225, 0.224, 0.229)

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # ── Stem: three separate 3×3 convs (ResNet-C style) ──────────────
        # Each conv has its own BN at the top level (NOT a Sequential).
        # Checkpoint keys:  conv1/bn1,  conv2/bn2,  conv3/bn3
        self.conv1   = nn.Conv2d(3,   64,  3, stride=2, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.conv2   = nn.Conv2d(64,  64,  3, stride=1, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64,  128, 3, stride=1, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(128)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ── Backbone (ResNet-101, dilated in layers 3 & 4) ───────────────
        self.layer1, c = _make_layer(128, 64,   3)
        self.layer2, c = _make_layer(c,   128,  4, stride=2)
        self.layer3, c = _make_layer(c,   256, 23, dilation=2,
                                     multi_grid=(1, 1, 1))
        self.layer4, c = _make_layer(c,   512,  3, dilation=4,
                                     multi_grid=(1, 2, 4))

        # ── Heads ─────────────────────────────────────────────────────────
        self.context_encoding = _PSPModule(2048, 512)
        self.edge    = _Edge()
        self.decoder = _Decoder(num_classes)

        # fushion takes layer3 (1024ch) → final parsing
        # fushion.3 is Conv with bias  →  use default bias=True
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),   # idx 0
            nn.BatchNorm2d(256),                    # idx 1
            nn.ReLU(inplace=True),                  # idx 2  (no params)
            nn.Conv2d(256, num_classes, 1),         # idx 3
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape

        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # Backbone
        l1 = self.layer1(x)    # 256ch
        l2 = self.layer2(l1)   # 512ch
        l3 = self.layer3(l2)   # 1024ch
        l4 = self.layer4(l3)   # 2048ch

        # Heads
        context = self.context_encoding(l4)
        edge, _ = self.edge(l1, l2, l3)
        parsing = self.decoder(context, l1)   # CE2P decoder — correct inference head
        _       = self.fushion(l3)            # training-only self-correction head (unused at test time)

        parsing = F.interpolate(parsing, (h, w), mode='bilinear', align_corners=True)
        edge    = F.interpolate(edge,    (h, w), mode='bilinear', align_corners=True)
        return parsing, edge


# ─── Loading ──────────────────────────────────────────────────────────────────

_SCHP_CHECKPOINT_HINT = (
    "Download the SCHP LIP checkpoint and save to  models/schp_lip.pth\n"
    "  URL: https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH\n"
    "  (LIP_final_Training_model.pth from the SCHP repo)"
)


def _default_ckpt_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "schp_lip.pth"


@lru_cache(maxsize=1)
def _load_schp(ckpt_path: str, device: str) -> AugmentCE2P:
    model = AugmentCE2P(num_classes=20)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_human(
    image: Image.Image,
    *,
    ckpt_path: str | Path | None = None,
    device: str | None = None,
    keypoints: np.ndarray | None = None,
    person_mask: np.ndarray | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Returns a (H, W) uint8 array with LIP label indices (0–19).

    image       — person PIL image at any resolution.
    target_size — (width, height) to resize the output labels to.
                  If None the output matches the input image size.
    If the SCHP checkpoint is not found, falls back to heuristic labels from
    keypoints + person_mask.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = Path(ckpt_path) if ckpt_path else _default_ckpt_path()

    if ckpt.exists():
        return _schp_parse(image, str(ckpt), device, target_size=target_size)

    warnings.warn(
        f"SCHP checkpoint not found at {ckpt}. Using heuristic fallback.\n"
        + _SCHP_CHECKPOINT_HINT,
        stacklevel=2,
    )
    labels = _heuristic_parse(image, keypoints=keypoints, person_mask=person_mask)
    if target_size is not None:
        labels = np.array(
            Image.fromarray(labels).resize(target_size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return labels


def _schp_parse(image: Image.Image, ckpt_path: str, device: str,
                target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Run SCHP inference with aspect-ratio-preserving resize + centre-crop.

    target_size — (width, height) for the output label map.
                  If None, output matches input image dimensions.
    """
    w0, h0 = image.size
    S = 473

    # Fit into S×S preserving aspect ratio; pad with ImageNet mean colour
    scale = S / max(w0, h0)
    new_w, new_h = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))

    pad_rgb = tuple(int(m * 255) for m in reversed(AugmentCE2P.MEAN))  # BGR→RGB
    canvas = Image.new("RGB", (S, S), pad_rgb)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    rgb = np.array(canvas).astype(np.float32) / 255.0
    bgr = rgb[:, :, ::-1].copy()
    mean = np.array(AugmentCE2P.MEAN, dtype=np.float32)
    std  = np.array(AugmentCE2P.STD,  dtype=np.float32)
    bgr  = (bgr - mean) / std
    inp  = torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    model = _load_schp(ckpt_path, device)
    with torch.no_grad():
        logits, _ = model(inp)

    labels_s    = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    labels_crop = labels_s[pad_y: pad_y + new_h, pad_x: pad_x + new_w]

    out_size = target_size if target_size is not None else (w0, h0)
    labels_img = Image.fromarray(labels_crop, mode="L")
    labels_img = labels_img.resize(out_size, Image.Resampling.NEAREST)
    return np.array(labels_img, dtype=np.uint8)


# ─── Heuristic fallback ───────────────────────────────────────────────────────

def _heuristic_parse(
    image: Image.Image,
    keypoints: np.ndarray | None,
    person_mask: np.ndarray | None,
) -> np.ndarray:
    w, h = image.size
    rgb = np.array(image)

    if person_mask is None:
        person_mask = _coarse_person_mask(rgb)
    if keypoints is None or np.all(keypoints[:, 2] == 0):
        keypoints = _geometry_keypoints(w, h, person_mask)

    label = np.zeros((h, w), dtype=np.uint8)
    kp = keypoints

    nose_y       = kp[0, 1]
    neck_y       = kp[1, 1]
    l_sho_y, r_sho_y = kp[5, 1], kp[2, 1]
    shoulder_y   = (l_sho_y + r_sho_y) / 2
    l_hip_y, r_hip_y = kp[11, 1], kp[8, 1]
    hip_y        = (l_hip_y + r_hip_y) / 2
    l_ank_y, r_ank_y = kp[13, 1], kp[10, 1]
    ankle_y      = (l_ank_y + r_ank_y) / 2

    l_sho_x, r_sho_x = kp[5, 0], kp[2, 0]
    l_hip_x, r_hip_x = kp[11, 0], kp[8, 0]
    l_elb_x, r_elb_x = kp[6, 0], kp[3, 0]
    l_wri_x, r_wri_x = kp[7, 0], kp[4, 0]

    cx = (l_sho_x + r_sho_x) / 2
    shoulder_half = max(16.0, abs(r_sho_x - l_sho_x) * 0.6)
    hip_half      = max(12.0, abs(r_hip_x - l_hip_x) * 0.6)

    torso = _polygon_mask(h, w, [
        (l_sho_x - shoulder_half * 0.2, shoulder_y),
        (r_sho_x + shoulder_half * 0.2, shoulder_y),
        (r_hip_x + hip_half * 0.2,      hip_y),
        (l_hip_x - hip_half * 0.2,      hip_y),
    ])

    arm_width = max(10.0, shoulder_half * 0.38)
    l_arm = _arm_mask(h, w,
                      (l_sho_x, l_sho_y), (l_elb_x, kp[6, 1]),
                      (l_wri_x, kp[7, 1]), arm_width)
    r_arm = _arm_mask(h, w,
                      (r_sho_x, r_sho_y), (r_elb_x, kp[3, 1]),
                      (r_wri_x, kp[4, 1]), arm_width)

    lower = np.zeros((h, w), dtype=np.uint8)
    hip_row = int(np.clip(hip_y, 0, h - 1))
    lower[hip_row:, :] = person_mask[hip_row:, :]

    sho_row = int(np.clip(shoulder_y, 0, h - 1))
    head_region = person_mask.copy()
    head_region[sho_row:, :] = 0

    l_eye, r_eye = kp[15], kp[14]
    if l_eye[2] > 0 and r_eye[2] > 0:
        eye_sep = abs(r_eye[0] - l_eye[0])
        face_cx = (l_eye[0] + r_eye[0]) / 2
        face_cy = (l_eye[1] + r_eye[1]) / 2 + eye_sep * 0.25
        face_rx = max(12, int(eye_sep * 0.90))
        face_ry = max(16, int(eye_sep * 1.30))
    else:
        face_cx = cx
        face_cy = nose_y + (neck_y - nose_y) * 0.3
        face_rx = max(12, int(shoulder_half * 0.48))
        face_ry = max(16, int(shoulder_half * 0.70))

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(face_mask, (int(face_cx), int(face_cy)),
                (face_rx, face_ry), 0, 0, 360, 255, -1)
    face_mask = cv2.bitwise_and(face_mask, head_region)
    hair_mask = cv2.bitwise_and(head_region, cv2.bitwise_not(face_mask))

    label[person_mask > 0] = 5
    label[torso > 0]       = 5
    label[l_arm > 0]       = 14
    label[r_arm > 0]       = 15
    label[lower > 0]       = 9
    label[hair_mask > 0]   = 2
    label[face_mask > 0]   = 13

    shoe_row = int(np.clip(ankle_y - h * 0.04, 0, h - 1))
    l_cx = int(np.clip(kp[13, 0], 0, w - 1))
    r_cx = int(np.clip(kp[10, 0], 0, w - 1))
    shoe_w = max(8, int(shoulder_half * 0.35))
    shoe_sub = label[shoe_row:, :]
    shoe_sub[:, max(0, l_cx - shoe_w):min(w, l_cx + shoe_w)] = 18
    shoe_sub[:, max(0, r_cx - shoe_w):min(w, r_cx + shoe_w)] = 19
    label[person_mask == 0] = 0
    return label


def _polygon_mask(h: int, w: int, pts: list) -> np.ndarray:
    from PIL import ImageDraw
    pil = Image.new("L", (w, h), 0)
    ImageDraw.Draw(pil).polygon(pts, fill=255)
    return np.array(pil, dtype=np.uint8)


def _arm_mask(h, w, p_shoulder, p_wrist, p_unused, width):
    pts = _thick_line_polygon(p_shoulder, p_wrist, width)
    return _polygon_mask(h, w, pts)


def _thick_line_polygon(a, b, half_w):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    length = max(1.0, (dx**2 + dy**2) ** 0.5)
    nx, ny = -dy / length * half_w, dx / length * half_w
    return [(ax+nx, ay+ny), (ax-nx, ay-ny), (bx-nx, by-ny), (bx+nx, by+ny)]


def _coarse_person_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    mask = (dist > 20).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8), iterations=2)
    return mask


def _geometry_keypoints(w, h, mask):
    ys, xs = np.where(mask > 0)
    if len(xs):
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        cx = (x1 + x2) / 2; bh = max(1, y2 - y1); bw = max(1, x2 - x1)
    else:
        cx, y1, bh, bw = w/2, 0, h, w
    sw = bw * 0.38
    pts = np.zeros((18, 3), dtype=np.float32)
    defs = {
        0: (cx, y1+bh*0.06), 1: (cx, y1+bh*0.13),
        2: (cx-sw, y1+bh*0.15), 3: (cx-sw*1.1, y1+bh*0.30),
        4: (cx-sw, y1+bh*0.44), 5: (cx+sw, y1+bh*0.15),
        6: (cx+sw*1.1, y1+bh*0.30), 7: (cx+sw, y1+bh*0.44),
        8: (cx-bw*0.12, y1+bh*0.55), 9: (cx-bw*0.12, y1+bh*0.72),
        10: (cx-bw*0.12, y1+bh*0.90), 11: (cx+bw*0.12, y1+bh*0.55),
        12: (cx+bw*0.12, y1+bh*0.72), 13: (cx+bw*0.12, y1+bh*0.90),
        14: (cx-bw*0.06, y1+bh*0.04), 15: (cx+bw*0.06, y1+bh*0.04),
        16: (cx-bw*0.12, y1+bh*0.06), 17: (cx+bw*0.12, y1+bh*0.06),
    }
    for i, (x, y) in defs.items():
        pts[i] = [np.clip(x, 0, w-1), np.clip(y, 0, h-1), 0.5]
    return pts
