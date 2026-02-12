"""SAM 3D Depth pipeline — extracts depth and normal maps from video frames.

Takes a single video frame, runs it through Meta's SAM 3D Objects
geometry model, and renders the resulting Gaussian geometry as a depth
map, surface normal map, or both side-by-side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .model_loader import load_sam3d_pipeline
from .renderers.depth import render_depth_map
from .renderers.normals import render_normal_map
from .schema import SAM3DConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class SAM3DPipeline(Pipeline):
    """Extract depth and normal maps using SAM 3D's geometry model."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return SAM3DConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.prev_result: torch.Tensor | None = None

        # Load-time params
        resolution = kwargs.get("inference_resolution", 512)
        num_steps = kwargs.get("num_inference_steps", 12)

        self.model = load_sam3d_pipeline(
            device=self.device,
            num_inference_steps=num_steps,
        )
        self._inference_resolution = resolution

    def prepare(self, **kwargs) -> Requirements:
        """We need exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process a single video frame through SAM 3D geometry.

        Accepts the frame, runs the geometry model, and renders the
        requested output (depth map, normal map, or both).
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("SAM3DPipeline requires video input")

        # Read runtime params from kwargs
        output_mode = kwargs.get("output_mode", "depth")
        depth_colormap = kwargs.get("depth_colormap", "magma")
        depth_invert = kwargs.get("depth_invert", False)
        normal_space = kwargs.get("normal_space", "camera")
        overlay_opacity = kwargs.get("overlay_opacity", 1.0)
        temporal_smoothing = kwargs.get("temporal_smoothing", 0.3)

        # Normalize input: (1, H, W, C) [0,255] -> (H, W, C) [0,1]
        frame = video[0].squeeze(0).to(
            device=self.device, dtype=torch.float32
        ) / 255.0
        H, W, _C = frame.shape

        # Resize to inference resolution if needed
        inf_res = self._inference_resolution
        if max(H, W) != inf_res:
            frame_resized = _resize_frame(frame, inf_res)
        else:
            frame_resized = frame

        # Generate full-frame mask (auto mode: all ones)
        rH, rW = frame_resized.shape[:2]
        mask = torch.ones(rH, rW, device=self.device, dtype=torch.float32)

        # Run SAM 3D geometry model
        with torch.no_grad():
            geo_output = self.model(frame_resized, mask)

        gaussians = geo_output["gaussians"]
        layout = {
            "rotation": geo_output.get("rotation"),
            "translation": geo_output.get("translation"),
            "scale": geo_output.get("scale"),
        }

        # Render requested output at original resolution
        result = None

        if output_mode in ("depth", "both"):
            depth = render_depth_map(
                gaussians=gaussians,
                layout=layout,
                output_size=(H, W),
                device=self.device,
                colormap=depth_colormap,
                invert=depth_invert,
            )

        if output_mode in ("normals", "both"):
            normals = render_normal_map(
                gaussians=gaussians,
                layout=layout,
                output_size=(H, W),
                device=self.device,
                space=normal_space,
            )

        if output_mode == "depth":
            result = depth
        elif output_mode == "normals":
            result = normals
        else:
            # Side-by-side: concatenate along width
            result = torch.cat([depth, normals], dim=1)

        # Temporal smoothing (EMA blend with previous frame)
        if temporal_smoothing > 0 and self.prev_result is not None:
            if self.prev_result.shape == result.shape:
                result = (
                    temporal_smoothing * self.prev_result
                    + (1 - temporal_smoothing) * result
                )
        self.prev_result = result.detach()

        # Overlay with original frame
        if overlay_opacity < 1.0:
            # For "both" mode, the result is wider than the original frame,
            # so we tile the original frame to match
            if result.shape[1] != W:
                original = torch.cat([frame, frame], dim=1)
            else:
                original = frame
            result = overlay_opacity * result + (1 - overlay_opacity) * original

        # Output in THWC format: add time dimension
        result = result.unsqueeze(0).clamp(0, 1)

        return {"video": result}


def _resize_frame(
    frame: torch.Tensor, target_size: int
) -> torch.Tensor:
    """Resize a (H, W, C) frame so the longest edge equals target_size.

    Preserves aspect ratio. Uses bilinear interpolation.
    """
    H, W, C = frame.shape
    if H >= W:
        new_h = target_size
        new_w = int(W * target_size / H)
    else:
        new_w = target_size
        new_h = int(H * target_size / W)

    # interpolate expects NCHW
    nchw = frame.permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        nchw, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).permute(1, 2, 0)
