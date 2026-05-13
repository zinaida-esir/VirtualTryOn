# VirtualTryOn

CP-VTON+ based virtual try-on prototype. The app accepts two images:

- a full-body person photo
- a product/flat-lay garment photo

The backend derives the VTON inputs automatically: human parsing, pose, body
shape, garment mask, agnostic representation, and then runs GMM + TOM.

## Runtime Path

The main backend flow is:

```text
backend/app.py
  -> tryon_pipeline.run_virtual_tryon()
  -> tryon_preprocess.prepare_tryon_inputs()
       - FASHN human parsing by default, mapped to CP-VTON+/LIP labels
       - optional SCHP LIP human parsing from models/schp_hf
       - Keypoint R-CNN pose estimation
       - OpenCV garment mask
       - CP-VTON+ tensors: image, head, shape, pose, cloth, cloth_mask
  -> tryon_infer.TryOnModel()
       - GMM warp
       - TOM render
```

Set `VTON_HUMAN_PARSER=schp` to use the packaged SCHP parser instead of FASHN.
The SCHP path is `tryon_parse_hf.py`, which loads the packaged local model from
`models/schp_hf`. The FASHN path is `tryon_parse_fashn.py`; on first use it may
download model weights through Hugging Face.

## Required Files

Keep these local model files:

```text
models/schp_hf/config.json
models/schp_hf/configuration_schp.py
models/schp_hf/image_processing_schp.py
models/schp_hf/model.safetensors
models/schp_hf/modeling_schp.py
models/schp_hf/preprocessor_config.json
```

CP-VTON+ checkpoints are also required. Either put them at:

```text
../cp-vton-plus/checkpoints/GMM/gmm_final.pth
../cp-vton-plus/checkpoints/TOM/tom_final.pth
```

or set:

```bash
export VTON_GMM_CKPT=/path/to/gmm_final.pth
export VTON_TOM_CKPT=/path/to/tom_final.pth
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run

Backend:

```bash
python backend/app.py
```

Frontend:

```bash
python -m http.server 8002 -d frontend
```

Then open:

```text
http://127.0.0.1:8002
```

The backend listens on:

```text
http://127.0.0.1:5051/api/tryon
```

## Debug Notebooks

Use these notebooks to inspect what the model receives:

- `tryon_test_from_dataset.ipynb`: loads a real CP-VTON+ dataset sample and
  visualizes the dataset tensors.
- `tryon_test_from_user_images.ipynb`: starts from `photo_full.png` and a
  garment image, derives masks/pose/parsing, and visualizes the exact tensors
  passed into `model.tryon()`.
- `parse_debug.ipynb`: shows raw SCHP labels and derived parsing masks before
  VTON inference.

For visual comparison, focus on:

```text
image
head
shape
pose_image
cloth
cloth_mask
parse_cloth
parse_cloth_mask
agnostic (max-channel visualization)
```

These should be compared against the same views from
`tryon_test_from_dataset.ipynb`.

## Notes

- CP-VTON+ expects `192x256` tensors. The preprocessing code keeps all generated
  labels and tensors in the same coordinate system.
- The `head` tensor is CP-VTON+ TOM's reserved region, not only the literal
  face/head. It preserves labels `{1, 2, 4, 9, 12, 13, 16, 17}`: hat, hair,
  sunglasses, pants, skirt, face, left leg, right leg.
- Jacket/blazer try-on is much harder than simple shirts because this checkpoint
  is sensitive to garment category, pose, and training distribution.
