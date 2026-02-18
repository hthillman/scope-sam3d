"""Render depth maps from SAM 3D Gaussian splat output.

The approach: project each Gaussian's center onto the image plane,
accumulate depth values weighted by opacity, and produce a per-pixel
depth map. This avoids needing a full differentiable Gaussian renderer
(gsplat) at render time — we only need positions and opacities.

For colormapping we use pure PyTorch (no matplotlib dependency).
"""

from __future__ import annotations

import torch


# Pre-built colormaps as (256, 3) tensors. Generated once, cached on device.
_COLORMAP_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _get_colormap(name: str, device: torch.device) -> torch.Tensor:
    """Get a (256, 3) colormap tensor on the given device."""
    cache_key = (name, str(device))
    if cache_key in _COLORMAP_CACHE:
        return _COLORMAP_CACHE[cache_key]

    t = torch.linspace(0.0, 1.0, 256, device=device)

    if name == "grayscale":
        cm = torch.stack([t, t, t], dim=-1)
    elif name == "magma":
        # Approximate magma colormap (dark purple -> orange -> yellow)
        r = torch.clamp(1.3 * t - 0.3, 0, 1) ** 0.8
        g = torch.clamp(0.7 * t ** 1.5, 0, 1)
        b = torch.clamp(0.5 + 0.8 * torch.sin(t * 3.14159 * 0.8), 0, 1) * (
            1.0 - t * 0.5
        )
        cm = torch.stack([r, g, b], dim=-1)
    elif name == "viridis":
        # Approximate viridis (dark purple -> teal -> yellow)
        r = torch.clamp(-0.3 + 1.5 * t ** 0.7, 0, 1)
        g = torch.clamp(0.1 + 0.9 * t, 0, 1) ** 0.9
        b = torch.clamp(0.6 - 0.4 * t + 0.3 * torch.sin(t * 3.14159), 0, 1)
        cm = torch.stack([r, g, b], dim=-1)
    elif name == "plasma":
        # Approximate plasma (dark blue -> magenta -> yellow)
        r = torch.clamp(0.05 + 1.2 * t ** 0.6, 0, 1)
        g = torch.clamp(-0.2 + 1.2 * t ** 1.5, 0, 1)
        b = torch.clamp(0.7 - 0.8 * t + 0.3 * torch.sin(t * 3.14), 0, 1)
        cm = torch.stack([r, g, b], dim=-1)
    else:
        # Fallback to grayscale
        cm = torch.stack([t, t, t], dim=-1)

    _COLORMAP_CACHE[cache_key] = cm
    return cm


def render_depth_from_gaussians(
    gaussians: object,
    layout: dict,
    output_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Render a raw depth map from Gaussian splat positions.

    Projects Gaussian centers onto the image plane using a simple
    orthographic projection (sufficient for depth visualization).
    Returns a (H, W) depth tensor normalized to [0, 1].

    Args:
        gaussians: Gaussian splat object with get_xyz and get_opacity.
        layout: Dict with "rotation", "translation", "scale" tensors.
        output_size: (H, W) of the output depth map.
        device: Torch device.

    Returns:
        (H, W) float32 tensor, [0, 1] range, where 0 = near, 1 = far.
    """
    H, W = output_size

    # Get Gaussian positions — shape (N, 3) in object space
    xyz = gaussians.get_xyz.detach().to(device)  # (N, 3)
    opacity = gaussians.get_opacity.detach().to(device).squeeze(-1)  # (N,)

    # Apply layout transform: rotate, scale, translate to camera space
    rotation = layout.get("rotation")
    translation = layout.get("translation")
    scale = layout.get("scale")

    if scale is not None:
        s = scale.to(device).view(1, 3)
        xyz = xyz * s

    if rotation is not None:
        rot_q = rotation.to(device).view(4)
        xyz = _quaternion_rotate(xyz, rot_q)

    if translation is not None:
        t = translation.to(device).view(1, 3)
        xyz = xyz + t

    # Orthographic projection: x, y map to pixel coords, z is depth
    # Normalize x, y from the Gaussian AABB to [0, W] and [0, H]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Normalize positions to [0, 1] range based on data bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    # Avoid division by zero
    x_range = (x_max - x_min).clamp(min=1e-6)
    y_range = (y_max - y_min).clamp(min=1e-6)
    z_range = (z_max - z_min).clamp(min=1e-6)

    px = ((x - x_min) / x_range * (W - 1)).long().clamp(0, W - 1)
    py = ((y - y_min) / y_range * (H - 1)).long().clamp(0, H - 1)
    depth_vals = (z - z_min) / z_range  # [0, 1]

    # Splat depth values onto the image using opacity-weighted accumulation
    depth_map = torch.zeros(H, W, device=device)
    weight_map = torch.zeros(H, W, device=device)

    # Use scatter_add for GPU-friendly accumulation
    flat_idx = py * W + px
    weighted_depth = depth_vals * opacity
    depth_map.view(-1).scatter_add_(0, flat_idx, weighted_depth)
    weight_map.view(-1).scatter_add_(0, flat_idx, opacity)

    # Normalize by accumulated weights
    valid = weight_map > 1e-6
    depth_map[valid] = depth_map[valid] / weight_map[valid]

    # Fill empty pixels with nearest-neighbor interpolation (simple dilation)
    depth_map = _fill_holes(depth_map, valid)

    return depth_map


def _quaternion_rotate(points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate points by a quaternion (wxyz convention).

    Args:
        points: (N, 3) tensor of points.
        q: (4,) quaternion in wxyz format.

    Returns:
        (N, 3) rotated points.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Build rotation matrix from quaternion
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    rot = torch.stack(
        [
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ]
    )  # (3, 3)

    return points @ rot.T


def _fill_holes(
    depth: torch.Tensor, valid: torch.Tensor, iterations: int = 5
) -> torch.Tensor:
    """Fill holes in a depth map by iterative dilation.

    Uses a 3x3 average of valid neighbors to fill empty pixels.
    """
    result = depth.clone()
    filled = valid.clone()

    for _ in range(iterations):
        # Pad for convolution
        padded = torch.nn.functional.pad(
            result.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        )
        padded_valid = torch.nn.functional.pad(
            filled.float().unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        )

        # 3x3 average filter
        kernel = torch.ones(1, 1, 3, 3, device=depth.device)
        neighbor_sum = torch.nn.functional.conv2d(padded, kernel).squeeze()
        neighbor_count = torch.nn.functional.conv2d(
            padded_valid, kernel
        ).squeeze()

        # Fill empty pixels that have valid neighbors
        can_fill = (~filled) & (neighbor_count > 0)
        if not can_fill.any():
            break

        result[can_fill] = neighbor_sum[can_fill] / neighbor_count[can_fill]
        filled = filled | can_fill

    return result


def render_depth_from_pointmap(
    pointmap: torch.Tensor,
    output_size: tuple[int, int],
    device: torch.device,
    colormap: str = "magma",
    invert: bool = False,
) -> torch.Tensor:
    """Render a colormapped depth map directly from a 3D pointmap.

    Extracts depth from the Z channel, normalizes to [0, 1], and applies
    a colormap. Much faster than the Gaussian splatting path.

    Args:
        pointmap: (3, H, W) XYZ coordinates in camera space.
        output_size: (H, W) to resize the output to.
        device: Torch device.
        colormap: One of "magma", "viridis", "plasma", "grayscale".
        invert: If True, near=dark and far=bright.

    Returns:
        (H, W, 3) float32 tensor in [0, 1] range.
    """
    depth = pointmap[2].to(device)  # (H_pm, W_pm)

    # Normalize to [0, 1] using only valid (non-zero) regions
    valid = depth.abs() > 1e-6
    if valid.any():
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        d_range = (d_max - d_min).clamp(min=1e-6)
        depth = ((depth - d_min) / d_range).clamp(0, 1)
        depth[~valid] = 0.0

    if invert:
        depth = 1.0 - depth

    # Resize to target resolution
    H, W = output_size
    if depth.shape[0] != H or depth.shape[1] != W:
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    # Apply colormap
    cm = _get_colormap(colormap, device)
    indices = (depth.clamp(0, 1) * 255).long().clamp(0, 255)
    return cm[indices]  # (H, W, 3)


def render_depth_map(
    gaussians: object,
    layout: dict,
    output_size: tuple[int, int],
    device: torch.device,
    colormap: str = "magma",
    invert: bool = False,
) -> torch.Tensor:
    """Render a colormapped depth map from Gaussian geometry.

    Args:
        gaussians: Gaussian splat object from SAM 3D.
        layout: Layout dict with rotation/translation/scale.
        output_size: (H, W) for the output image.
        device: Torch device.
        colormap: One of "magma", "viridis", "plasma", "grayscale".
        invert: If True, near=dark and far=bright.

    Returns:
        (H, W, 3) float32 tensor in [0, 1] range.
    """
    depth = render_depth_from_gaussians(gaussians, layout, output_size, device)

    if invert:
        depth = 1.0 - depth

    # Apply colormap
    cm = _get_colormap(colormap, device)
    indices = (depth.clamp(0, 1) * 255).long().clamp(0, 255)
    colored = cm[indices]  # (H, W, 3)

    return colored
