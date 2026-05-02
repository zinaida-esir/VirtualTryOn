# coding: utf-8
"""High-level entry point: two image paths → output JPEG."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from tryon_infer import TryOnModel
from tryon_preprocess import prepare_tryon_inputs, tensor_to_pil

# ─── Checkpoint discovery ────────────────────────────────────────────────────

def _find_checkpoints() -> tuple[Path, Path]:
    """
    Search common locations for gmm_final.pth / tom_final.pth.
    Looks in:
      1. <repo_root>/checkpoints/GMM|TOM/
      2. <repo_root>/../cp-vton-plus/checkpoints/GMM|TOM/
      3. ~/projects/cp-vton-plus/checkpoints/GMM|TOM/
      4. VTON_GMM_CKPT / VTON_TOM_CKPT environment variables
    """
    import os
    env_gmm = os.environ.get("VTON_GMM_CKPT")
    env_tom = os.environ.get("VTON_TOM_CKPT")
    if env_gmm and env_tom:
        return Path(env_gmm), Path(env_tom)

    here = Path(__file__).resolve().parent
    candidates = [
        here / "checkpoints",
        here.parent / "cp-vton-plus" / "checkpoints",
        here.parents[1] / "cp-vton-plus" / "checkpoints",
        here.parents[2] / "cp-vton-plus" / "checkpoints",
        Path.home() / "projects" / "cp-vton-plus" / "checkpoints",
    ]
    for base in candidates:
        gmm = base / "GMM" / "gmm_final.pth"
        tom = base / "TOM" / "tom_final.pth"
        if gmm.exists() and tom.exists():
            return gmm, tom

    raise FileNotFoundError(
        "CP-VTON+ checkpoints not found.\n"
        "Set environment variables VTON_GMM_CKPT and VTON_TOM_CKPT, or place them at:\n"
        "  <project>/../cp-vton-plus/checkpoints/GMM/gmm_final.pth\n"
        "  <project>/../cp-vton-plus/checkpoints/TOM/tom_final.pth"
    )


class TryOnPipelineError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _model() -> TryOnModel:
    gmm, tom = _find_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TryOnModel(str(gmm), str(tom), device=device, use_fp16=False)


# ─── Public API ──────────────────────────────────────────────────────────────

def run_virtual_tryon(
    person_path:  str | Path,
    cloth_path:   str | Path,
    output_path:  str | Path,
    *,
    debug_dir:    str | Path | None = None,
    device:       str | None = None,
) -> Path:
    """
    Run full virtual try-on pipeline.

    Parameters
    ----------
    person_path : full-body person photo (any resolution / format)
    cloth_path  : flat-lay garment image (any resolution / format)
    output_path : where to write the result JPEG
    debug_dir   : if given, saves intermediate images here for inspection
    device      : 'cpu' / 'cuda' / None (auto)

    Returns
    -------
    Path to the saved output image.
    """
    output_path = Path(output_path)
    try:
        sample = prepare_tryon_inputs(
            person_path,
            cloth_path,
            device=device,
        )

        if debug_dir is not None:
            _save_debug(sample, Path(debug_dir))

        model = _model()
        out = model.tryon(
            sample.agnostic.unsqueeze(0),
            sample.cloth.unsqueeze(0),
            sample.cloth_mask.unsqueeze(0),
        )
        result = tensor_to_pil(out)
        result.save(output_path, quality=92)
        return output_path

    except Exception as exc:
        raise TryOnPipelineError(str(exc)) from exc


def _save_debug(sample, debug_path: Path) -> None:
    import torchvision.transforms.functional as TF
    debug_path.mkdir(parents=True, exist_ok=True)
    sample.person_preview.save(debug_path / "debug_person.jpg", quality=90)
    sample.cloth_preview.save(debug_path / "debug_cloth.jpg", quality=90)
    sample.cloth_mask_preview.save(debug_path / "debug_cloth_mask.png")
    sample.person_mask_preview.save(debug_path / "debug_person_mask.png")

    def _save_tensor(t: "torch.Tensor", name: str) -> None:
        from PIL import Image as _Image
        if t.dim() == 3 and t.shape[0] in (1, 3):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            _Image.fromarray(arr).save(debug_path / name)

    _save_tensor(sample.head,   "debug_head.jpg")
    _save_tensor(sample.shape,  "debug_shape.png")
    _save_tensor(sample.pose_image, "debug_pose.png")
    _save_tensor(sample.parse_cloth_mask, "debug_parse_cloth_mask.png")
