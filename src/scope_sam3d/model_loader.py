"""Load SAM 3D Objects geometry pipeline.

The full SAM 3D pipeline has two stages:
  1. Sparse Structure Generation (ss_generator + ss_decoder) — 3D occupancy
  2. Dense Geometry Generation (slat_generator + decoders) — Gaussians/mesh

For depth and normal maps we need both stages: stage 1 gives us the coarse
voxel structure, and stage 2 gives us dense Gaussians from which we can
render high-quality depth and normals.

Model weights are gated on HuggingFace. The user must:
  1. Accept the SAM License at https://huggingface.co/facebook/sam-3d-objects
  2. Set HF_TOKEN env var or run `huggingface-cli login`
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

# Surface Hydra's chained exceptions so we see the real import error
os.environ["HYDRA_FULL_ERROR"] = "1"

logger = logging.getLogger(__name__)

# HuggingFace repo for SAM 3D Objects
HF_REPO_ID = "facebook/sam-3d-objects"


def _find_or_download_checkpoint(cache_dir: Path | None = None) -> Path:
    """Download SAM 3D checkpoint from HuggingFace if not already cached.

    Returns the path to the directory containing pipeline.yaml and model
    weights.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download SAM 3D weights. "
            "Install it with: pip install huggingface-hub"
        ) from exc

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "scope-sam3d"

    token = os.environ.get("HF_TOKEN")

    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            cache_dir=str(cache_dir),
            token=token,
            local_dir=str(cache_dir / "sam-3d-objects"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SAM 3D weights from {HF_REPO_ID}. "
            "Make sure you've accepted the SAM License at "
            f"https://huggingface.co/facebook/sam-3d-objects and set "
            "HF_TOKEN or run `huggingface-cli login`.\n"
            f"Error: {exc}"
        ) from exc

    return Path(local_dir)


def _find_config(checkpoint_dir: Path) -> Path:
    """Locate pipeline.yaml in the downloaded checkpoint directory.

    The HuggingFace repo has it at checkpoints/pipeline.yaml.
    """
    candidates = [
        checkpoint_dir / "checkpoints" / "pipeline.yaml",
        checkpoint_dir / "pipeline.yaml",
        checkpoint_dir / "checkpoints" / "hf" / "pipeline.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find pipeline.yaml in {checkpoint_dir}. "
        f"Searched: {[str(c) for c in candidates]}. "
        "The SAM 3D checkpoint may be incomplete."
    )


def load_sam3d_pipeline(
    device: torch.device,
    num_inference_steps: int = 12,
    cache_dir: Path | None = None,
) -> SAM3DInferenceWrapper:
    """Load the SAM 3D Objects inference pipeline.

    This loads the full pipeline from the sam3d_objects package, using
    Hydra/OmegaConf config to instantiate all model components. We then
    wrap it in a thin interface that accepts (image, mask) and returns
    Gaussian geometry for rendering.

    Args:
        device: Torch device (cuda recommended).
        num_inference_steps: Diffusion steps for geometry generation.
        cache_dir: Where to cache downloaded weights.

    Returns:
        A SAM3DInferenceWrapper with a __call__(image, mask) interface.
    """
    checkpoint_dir = _find_or_download_checkpoint(cache_dir)

    logger.info("Loading SAM 3D Objects pipeline from %s", checkpoint_dir)

    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
    except ImportError as exc:
        raise ImportError(
            "omegaconf and hydra-core are required for SAM 3D. "
            "They should be installed with the sam3d-objects package."
        ) from exc

    config_path = _find_config(checkpoint_dir)
    logger.info("Using config: %s", config_path)

    config = OmegaConf.load(str(config_path))

    # workspace_dir must point to the directory containing pipeline.yaml;
    # all model weight paths in the config are resolved relative to this.
    config.workspace_dir = str(config_path.parent)

    # Match the reference Inference class settings
    config.rendering_engine = "pytorch3d"
    config.compile_model = False

    # Try to import the target class directly first to get a clear error
    # if something in the import chain fails (Hydra swallows the real cause)
    try:
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa: F401
        logger.info("Successfully imported InferencePipelinePointMap")
    except Exception:
        import traceback
        logger.error(
            "Failed to import InferencePipelinePointMap. Full traceback:\n%s",
            traceback.format_exc(),
        )
        raise

    pipeline = instantiate(config)

    logger.info("SAM 3D pipeline loaded successfully on %s", device)

    return SAM3DInferenceWrapper(
        pipeline=pipeline,
        device=device,
        num_inference_steps=num_inference_steps,
    )


class SAM3DInferenceWrapper:
    """Thin wrapper around the SAM 3D inference pipeline.

    Accepts a normalized image tensor and binary mask, runs the geometry
    stage, and returns Gaussian splat parameters + layout for rendering.
    """

    def __init__(
        self,
        pipeline: object,
        device: torch.device,
        num_inference_steps: int = 12,
    ):
        self.pipeline = pipeline
        self.device = device
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int | None = None,
    ) -> dict:
        """Run SAM 3D geometry inference.

        Args:
            image: RGB image tensor (H, W, 3) in [0, 1] range, float32.
            mask: Binary mask tensor (H, W) in {0, 1}, float32.
            seed: Optional random seed for reproducibility.

        Returns:
            dict with keys:
                "gaussians": Gaussian splat object with xyz, opacity,
                             scaling, rotation, and features.
                "rotation": (4,) quaternion for object orientation.
                "translation": (3,) position in world frame.
                "scale": (3,) scale factors.
        """
        import numpy as np

        # Convert to RGBA uint8 numpy array (SAM 3D's expected input format)
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        rgba = np.concatenate(
            [img_np[..., :3], mask_np[..., None]], axis=-1
        )  # (H, W, 4)

        # Run pipeline using the .run() API (matches notebook/inference.py)
        output = self.pipeline.run(
            rgba,
            None,  # masks — handled internally
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
        )

        # Extract Gaussian geometry from pipeline output
        gs = output.get("gs")
        if gs is None:
            gs = output.get("gaussian")
            if isinstance(gs, list) and len(gs) > 0:
                gs = gs[0]

        # Layout info (rotation, translation, scale) is in the output
        # directly or nested under a "layout" key
        layout = output.get("layout", {})

        result = {
            "gaussians": gs,
            "rotation": output.get("rotation") or layout.get("rotation"),
            "translation": output.get("translation") or layout.get("translation"),
            "scale": output.get("scale") or layout.get("scale"),
        }

        return result
