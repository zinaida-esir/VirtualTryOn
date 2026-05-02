# coding: utf-8
"""
Pose estimation via torchvision KeypointRCNN.

Returns 18 keypoints in OpenPose COCO order (as used by CP-VTON+):
  0=Nose  1=Neck  2=RShoulder  3=RElbow  4=RWrist
  5=LShoulder  6=LElbow  7=LWrist
  8=RHip  9=RKnee  10=RAnkle  11=LHip  12=LKnee  13=LAnkle
  14=REye  15=LEye  16=REar  17=LEar

torchvision COCO keypoint order (17 points):
  0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
  5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
  9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
  13=left_knee 14=right_knee 15=left_ankle 16=right_ankle
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# COCO idx → OpenPose idx mapping (17-point COCO → 18-point OpenPose)
# -1 means synthesised (neck = mean of left/right shoulder)
_COCO_TO_OP: list[int | None] = [
    0,    # OP 0 nose       ← COCO 0
    None, # OP 1 neck       ← mean(COCO 5, COCO 6)
    6,    # OP 2 RShoulder  ← COCO 6
    8,    # OP 3 RElbow     ← COCO 8
    10,   # OP 4 RWrist     ← COCO 10
    5,    # OP 5 LShoulder  ← COCO 5
    7,    # OP 6 LElbow     ← COCO 7
    9,    # OP 7 LWrist     ← COCO 9
    12,   # OP 8 RHip       ← COCO 12
    14,   # OP 9 RKnee      ← COCO 14
    16,   # OP 10 RAnkle    ← COCO 16
    11,   # OP 11 LHip      ← COCO 11
    13,   # OP 12 LKnee     ← COCO 13
    15,   # OP 13 LAnkle    ← COCO 15
    2,    # OP 14 REye      ← COCO 2
    1,    # OP 15 LEye      ← COCO 1
    4,    # OP 16 REar      ← COCO 4
    3,    # OP 17 LEar      ← COCO 3
]


@lru_cache(maxsize=1)
def _load_model(device: str):
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()
    return model


def estimate_pose(
    image: Image.Image,
    device: str | None = None,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """
    Returns (18, 3) float32 array  [x, y, confidence]
    in pixel coordinates of `image`.
    If no person is detected, keypoints are heuristically estimated from
    the image dimensions.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_model(device)
    img_t = TF.to_tensor(image).to(device)

    with torch.no_grad():
        preds = model([img_t])

    w, h = image.size
    pred = preds[0]

    # pick highest-score person above threshold
    scores = pred["scores"].cpu().numpy()
    kp_all = pred["keypoints"].cpu().numpy()   # (N, 17, 3)  x, y, visibility

    best_idx = -1
    for i, s in enumerate(scores):
        if s >= score_threshold and (best_idx == -1 or s > scores[best_idx]):
            best_idx = i

    if best_idx == -1:
        return _fallback_pose(w, h)

    coco_kp = kp_all[best_idx]  # (17, 3)  x, y, visibility (0/1/2)

    op_kp = np.zeros((18, 3), dtype=np.float32)
    for op_i, coco_i in enumerate(_COCO_TO_OP):
        if coco_i is None:
            # neck = midpoint of shoulders
            ls = coco_kp[5]
            rs = coco_kp[6]
            if ls[2] > 0 and rs[2] > 0:
                op_kp[op_i] = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, 1.0]
            elif ls[2] > 0:
                op_kp[op_i] = [ls[0], ls[1], 0.5]
            elif rs[2] > 0:
                op_kp[op_i] = [rs[0], rs[1], 0.5]
        else:
            x, y, vis = coco_kp[coco_i]
            # visibility: 0=not labelled, 1=labelled but occluded, 2=visible
            conf = float(vis) / 2.0
            op_kp[op_i] = [x, y, conf]

    return op_kp


def _fallback_pose(w: int, h: int) -> np.ndarray:
    """Geometric estimate when no person is detected."""
    cx = w / 2
    pts = np.zeros((18, 3), dtype=np.float32)
    coords = {
        0:  (cx,        h * 0.08),   # nose
        1:  (cx,        h * 0.14),   # neck
        2:  (cx - w*0.19, h * 0.16), # RShoulder
        3:  (cx - w*0.21, h * 0.30), # RElbow
        4:  (cx - w*0.20, h * 0.44), # RWrist
        5:  (cx + w*0.19, h * 0.16), # LShoulder
        6:  (cx + w*0.21, h * 0.30), # LElbow
        7:  (cx + w*0.20, h * 0.44), # LWrist
        8:  (cx - w*0.11, h * 0.52), # RHip
        9:  (cx - w*0.11, h * 0.70), # RKnee
        10: (cx - w*0.11, h * 0.88), # RAnkle
        11: (cx + w*0.11, h * 0.52), # LHip
        12: (cx + w*0.11, h * 0.70), # LKnee
        13: (cx + w*0.11, h * 0.88), # LAnkle
        14: (cx - w*0.05, h * 0.06), # REye
        15: (cx + w*0.05, h * 0.06), # LEye
        16: (cx - w*0.10, h * 0.08), # REar
        17: (cx + w*0.10, h * 0.08), # LEar
    }
    for i, (x, y) in coords.items():
        pts[i] = [np.clip(x, 0, w-1), np.clip(y, 0, h-1), 0.5]
    return pts
