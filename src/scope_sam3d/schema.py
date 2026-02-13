from typing import ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class SAM3DConfig(BasePipelineConfig):
    """Configuration for SAM 3D depth/normal pipeline."""

    pipeline_id: ClassVar[str] = "sam3d-depth"
    pipeline_name: ClassVar[str] = "SAM 3D Depth"
    pipeline_description: ClassVar[str] = (
        "Extract depth maps and surface normals from video using "
        "Meta's SAM 3D Objects geometry model"
    )

    supports_prompts: ClassVar[bool] = False
    modified: ClassVar[bool] = True
    usage: ClassVar[list[UsageType]] = [UsageType.GENERATOR]
    estimated_vram_gb: ClassVar[float | None] = 9.0

    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    # --- Output Mode ---

    output_mode: Literal["depth", "normals", "both"] = Field(
        default="depth",
        description="What to render: depth map, normal map, or both side-by-side",
        json_schema_extra=ui_field_config(order=1, label="Output"),
    )

    # --- Depth Visualization ---

    depth_colormap: Literal["magma", "viridis", "plasma", "grayscale"] = Field(
        default="magma",
        description="Colormap for depth visualization",
        json_schema_extra=ui_field_config(order=10, label="Depth Colormap"),
    )

    depth_invert: bool = Field(
        default=False,
        description=(
            "Invert depth (near=dark, far=bright instead of default near=bright)"
        ),
        json_schema_extra=ui_field_config(order=11, label="Invert Depth"),
    )

    # --- Normal Map ---

    normal_space: Literal["camera", "world"] = Field(
        default="camera",
        description=(
            "Coordinate space for normal map rendering. "
            "Camera space is more useful for relighting; "
            "world space preserves object orientation."
        ),
        json_schema_extra=ui_field_config(order=20, label="Normal Space"),
    )

    # --- Overlay ---

    overlay_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Blend depth/normal output with original frame "
            "(0 = original, 1 = full effect)"
        ),
        json_schema_extra=ui_field_config(order=30, label="Effect Opacity"),
    )

    # --- Performance ---

    inference_resolution: int = Field(
        default=512,
        ge=256,
        le=1024,
        description=(
            "Resolution to run inference at (lower = faster, higher = more detail). "
            "Changing this requires a pipeline reload."
        ),
        json_schema_extra=ui_field_config(
            order=40,
            label="Inference Resolution",
            is_load_param=True,
        ),
    )

    temporal_smoothing: float = Field(
        default=0.3,
        ge=0.0,
        le=0.95,
        description=(
            "Blend current depth with previous frame to reduce flicker "
            "(0 = off, higher = smoother)"
        ),
        json_schema_extra=ui_field_config(order=41, label="Temporal Smoothing"),
    )

    num_inference_steps: int = Field(
        default=12,
        ge=1,
        le=50,
        description=(
            "Number of diffusion steps for geometry generation. "
            "Fewer steps = faster but lower quality. "
            "Changing this requires a pipeline reload."
        ),
        json_schema_extra=ui_field_config(
            order=42,
            label="Diffusion Steps",
            is_load_param=True,
        ),
    )
