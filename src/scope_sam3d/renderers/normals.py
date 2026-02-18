"""Render surface normal maps from SAM 3D Gaussian splat output.

Normal maps encode surface orientation as RGB color:
  R = normal.x, G = normal.y, B = normal.z
mapped from [-1, 1] to [0, 1] for display.

The approach: estimate surface normals from the local distribution of
Gaussian centers, then splat them onto the image plane weighted by
opacity. This gives a smooth, physically meaningful normal map without
requiring mesh extraction.
"""

from __future__ import annotations

import torch

from .depth import _quaternion_rotate


def _estimate_gaussian_normals(
    xyz: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Estimate surface normals from point positions via PCA of local neighborhoods.

    For each Gaussian center, find k nearest neighbors and compute the
    normal as the eigenvector corresponding to the smallest eigenvalue
    of the local covariance matrix. This is equivalent to fitting a
    local tangent plane.

    Args:
        xyz: (N, 3) point positions.
        k: Number of nearest neighbors for local plane estimation.

    Returns:
        (N, 3) unit normal vectors.
    """
    N = xyz.shape[0]
    device = xyz.device

    if N < k + 1:
        # Not enough points; return z-facing normals as fallback
        normals = torch.zeros(N, 3, device=device)
        normals[:, 2] = 1.0
        return normals

    # Compute pairwise distances (chunked for memory efficiency)
    chunk_size = min(4096, N)
    normals = torch.zeros(N, 3, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = xyz[start:end]  # (C, 3)

        # Distances from chunk points to all points
        dists = torch.cdist(chunk, xyz)  # (C, N)

        # Get k+1 nearest (includes self), take indices 1..k
        _, nn_idx = dists.topk(k + 1, dim=1, largest=False)
        nn_idx = nn_idx[:, 1:]  # (C, k)

        # Gather neighbor positions
        neighbors = xyz[nn_idx]  # (C, k, 3)

        # Center the neighborhoods
        centered = neighbors - chunk.unsqueeze(1)  # (C, k, 3)

        # Covariance matrix for each point
        cov = torch.bmm(centered.transpose(1, 2), centered) / k  # (C, 3, 3)

        # Eigendecomposition — smallest eigenvector is the normal
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # sorted ascending
        chunk_normals = eigenvectors[:, :, 0]  # (C, 3) — smallest eigenvalue

        # Orient normals consistently (pointing toward camera, i.e., -z)
        # Flip if the normal points away from the viewpoint
        flip = (chunk_normals[:, 2] > 0).float() * -2.0 + 1.0
        chunk_normals = chunk_normals * flip.unsqueeze(-1)

        normals[start:end] = chunk_normals

    # Normalize to unit length
    normals = torch.nn.functional.normalize(normals, dim=-1)

    return normals


def render_normal_map(
    gaussians: object,
    layout: dict,
    output_size: tuple[int, int],
    device: torch.device,
    space: str = "camera",
) -> torch.Tensor:
    """Render a surface normal map from Gaussian geometry.

    Args:
        gaussians: Gaussian splat object from SAM 3D with get_xyz, get_opacity.
        layout: Layout dict with rotation/translation/scale.
        output_size: (H, W) for the output image.
        device: Torch device.
        space: "camera" or "world". Camera space transforms normals by
               the layout rotation; world space preserves object-space
               orientation.

    Returns:
        (H, W, 3) float32 tensor in [0, 1] range encoding normals.
        RGB = (nx+1)/2, (ny+1)/2, (nz+1)/2.
    """
    H, W = output_size

    # Get Gaussian positions and opacities
    xyz = gaussians.get_xyz.detach().to(device)  # (N, 3)
    opacity = gaussians.get_opacity.detach().to(device).squeeze(-1)  # (N,)

    # Apply layout transform to positions (for projection)
    xyz_transformed = xyz.clone()
    rotation = layout.get("rotation")
    translation = layout.get("translation")
    scale = layout.get("scale")

    if scale is not None:
        s = scale.to(device).view(1, 3)
        xyz_transformed = xyz_transformed * s

    if rotation is not None:
        rot_q = rotation.to(device).view(4)
        xyz_transformed = _quaternion_rotate(xyz_transformed, rot_q)

    if translation is not None:
        t = translation.to(device).view(1, 3)
        xyz_transformed = xyz_transformed + t

    # Estimate normals from point cloud
    normals = _estimate_gaussian_normals(xyz, k=min(8, max(3, xyz.shape[0] // 10)))

    # Transform normals to camera space if requested
    if space == "camera" and rotation is not None:
        rot_q = rotation.to(device).view(4)
        normals = _quaternion_rotate(normals, rot_q)

    # Project positions onto image plane (orthographic)
    x, y = xyz_transformed[:, 0], xyz_transformed[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = (x_max - x_min).clamp(min=1e-6)
    y_range = (y_max - y_min).clamp(min=1e-6)

    px = ((x - x_min) / x_range * (W - 1)).long().clamp(0, W - 1)
    py = ((y - y_min) / y_range * (H - 1)).long().clamp(0, H - 1)

    # Splat normals onto image with opacity weighting
    normal_map = torch.zeros(H, W, 3, device=device)
    weight_map = torch.zeros(H, W, device=device)

    flat_idx = py * W + px

    for c in range(3):
        weighted_n = normals[:, c] * opacity
        normal_map[:, :, c].view(-1).scatter_add_(0, flat_idx, weighted_n)

    weight_map.view(-1).scatter_add_(0, flat_idx, opacity)

    # Normalize by weights
    valid = weight_map > 1e-6
    for c in range(3):
        normal_map[:, :, c][valid] = (
            normal_map[:, :, c][valid] / weight_map[valid]
        )

    # Re-normalize to unit length after splatting
    norms = normal_map.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normal_map = normal_map / norms

    # Fill holes (simple dilation for empty pixels)
    normal_map = _fill_normal_holes(normal_map, valid)

    # Map from [-1, 1] to [0, 1] for display
    normal_map = (normal_map + 1.0) * 0.5

    return normal_map.clamp(0, 1)


def render_normals_from_pointmap(
    pointmap: torch.Tensor,
    output_size: tuple[int, int],
    device: torch.device,
    space: str = "camera",
) -> torch.Tensor:
    """Render a normal map from spatial gradients of a 3D pointmap.

    Computes normals as the cross product of horizontal and vertical
    gradients in the XYZ pointmap. Much faster than PCA-based estimation
    from Gaussian point clouds.

    Args:
        pointmap: (3, H, W) XYZ coordinates in camera space.
        output_size: (H, W) to resize the output to.
        device: Torch device.
        space: "camera" (default). Unused here since the pointmap is
               already in camera space, but kept for API compatibility.

    Returns:
        (H, W, 3) float32 tensor in [0, 1] range encoding normals.
        RGB = (nx+1)/2, (ny+1)/2, (nz+1)/2.
    """
    pm = pointmap.to(device)  # (3, H_pm, W_pm)

    # Spatial gradients: dx along width, dy along height
    # pointmap shape is (3, H, W), so dim=2 is width, dim=1 is height
    dx = pm[:, :, 1:] - pm[:, :, :-1]  # (3, H, W-1)
    dy = pm[:, 1:, :] - pm[:, :-1, :]  # (3, H-1, W)

    # Trim to common size: (3, H-1, W-1)
    dx = dx[:, :-1, :]  # (3, H-1, W-1)
    dy = dy[:, :, :-1]  # (3, H-1, W-1)

    # Cross product: normal = dy × dx (right-hand rule, camera-facing)
    # Permute to (H-1, W-1, 3) for the cross product
    dx_hwc = dx.permute(1, 2, 0)  # (H-1, W-1, 3)
    dy_hwc = dy.permute(1, 2, 0)  # (H-1, W-1, 3)

    normals = torch.cross(dy_hwc, dx_hwc, dim=-1)  # (H-1, W-1, 3)

    # Normalize to unit length
    norms = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / norms

    # Orient normals toward camera (negative z in camera space)
    # Flip normals that point away from camera
    flip = (normals[..., 2] > 0).float() * -2.0 + 1.0
    normals = normals * flip.unsqueeze(-1)

    # Resize to target resolution
    H, W = output_size
    if normals.shape[0] != H or normals.shape[1] != W:
        # Interpolate in CHW format
        normals_chw = normals.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H-1, W-1)
        normals_chw = torch.nn.functional.interpolate(
            normals_chw, size=(H, W), mode="bilinear", align_corners=False
        )
        normals = normals_chw.squeeze(0).permute(1, 2, 0)  # (H, W, 3)

        # Re-normalize after interpolation
        norms = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms

    # Map from [-1, 1] to [0, 1] for display
    normal_map = (normals + 1.0) * 0.5

    return normal_map.clamp(0, 1)


def _fill_normal_holes(
    normals: torch.Tensor, valid: torch.Tensor, iterations: int = 5
) -> torch.Tensor:
    """Fill holes in a normal map via iterative dilation.

    Averages valid neighbor normals and re-normalizes.
    """
    result = normals.clone()
    filled = valid.clone()

    for _ in range(iterations):
        # Process each channel with a 3x3 average
        padded = torch.nn.functional.pad(
            result.permute(2, 0, 1).unsqueeze(0),
            (1, 1, 1, 1),
            mode="replicate",
        )
        padded_valid = torch.nn.functional.pad(
            filled.float().unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        )

        kernel = torch.ones(1, 1, 3, 3, device=normals.device)

        # Sum of valid neighbor normals per channel
        neighbor_count = torch.nn.functional.conv2d(
            padded_valid, kernel
        ).squeeze(0).squeeze(0)

        can_fill = (~filled) & (neighbor_count > 0)
        if not can_fill.any():
            break

        for c in range(3):
            channel = padded[:, c : c + 1, :, :]
            neighbor_sum = torch.nn.functional.conv2d(
                channel, kernel
            ).squeeze()
            result[:, :, c][can_fill] = (
                neighbor_sum[can_fill] / neighbor_count[can_fill]
            )

        # Re-normalize filled normals
        norms = result[can_fill].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        result[can_fill] = result[can_fill] / norms

        filled = filled | can_fill

    return result
