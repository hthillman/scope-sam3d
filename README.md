# scope-sam3d

Depth and normal maps from Meta's SAM 3D Objects geometry model, built as a [Daydream Scope](https://github.com/daydreamlive/scope) preprocessor plugin.

Runs a single video frame through SAM 3D's MoGe monocular depth estimator (a single forward pass through a 1.2B parameter model), then derives per-pixel depth and surface normal maps directly from the resulting 3D pointmap. No mesh extraction, no Gaussian splatting, no multi-step diffusion; the 3D geometry is computed in one shot and projected back into 2D.

## Why depth and normal maps?

**Depth maps** encode how far each pixel is from the camera. Useful for:

- **Parallax and 3D photo effects** — shift layers at different speeds based on distance to create a sense of depth from a flat video frame
- **Synthetic depth of field** — blur the background (or foreground) based on depth, mimicking a shallow-focus lens
- **Occlusion-aware compositing** — place virtual objects into a scene and have them correctly appear behind real-world geometry
- **Fog and atmospheric effects** — apply distance-based haze, fog, or color grading that respects scene geometry

**Normal maps** encode the orientation of each surface (which direction it faces). Useful for:

- **Relighting** — apply virtual light sources to a real scene and have surfaces respond correctly based on their angle to the light
- **Surface-aware effects** — edge detection, material segmentation, and stylization that follows actual surface geometry rather than just color boundaries
- **Bump and detail enhancement** — exaggerate surface detail for stylized looks by manipulating the normal map before rendering

**Both (side-by-side)** shows depth and normals together, which is useful for debugging geometry quality or comparing the two representations on the same frame.

## Requirements

- NVIDIA GPU with 12GB+ VRAM (tested on RTX 4090, A100)
- Accept the [SAM License](https://huggingface.co/facebook/sam-3d-objects) on HuggingFace
- Set `HF_TOKEN` environment variable or run `huggingface-cli login`

## Installation

In Scope, go to **Settings > Plugins** and enter one of:

**Git URL:**
```
git+https://github.com/hthillman/scope-sam3d.git
```

**Local path (development):**
```
/path/to/scope-sam3d
```

First load will download ~5GB of model weights from HuggingFace.

## Usage

1. Select **SAM 3D Depth** from the preprocessor list
2. Choose an **Output** mode:
   - **depth** — colormapped depth map (default)
   - **normals** — RGB-encoded surface normal map
   - **both** — depth and normals side-by-side
3. Adjust **Effect Opacity** to blend with the original frame, or leave at 1.0 for the full effect
4. Use **Temporal Smoothing** to reduce per-frame flicker (0.3 is a good default; increase for smoother output at the cost of temporal lag)

### Tuning depth output

- **Depth Colormap** — choose between magma (default, purple-to-yellow), viridis (purple-to-green-to-yellow), plasma (blue-to-magenta-to-yellow), or grayscale
- **Invert Depth** — by default, near objects are bright and far objects are dark. Toggle this to flip that mapping.

### Tuning normal output

- **Normal Space** — "camera" (default) orients normals relative to the viewer, which is what you want for relighting. "World" preserves the object's intrinsic surface orientation regardless of camera angle.

## Parameters

### Load-time (require pipeline reload)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Inference Resolution | 512 | 256-1024 | Resolution for MoGe inference. Lower is faster, higher captures finer geometry. |

### Runtime (adjustable during streaming)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Output | depth | depth, normals, both | What to render |
| Depth Colormap | magma | magma, viridis, plasma, grayscale | Colormap for depth visualization |
| Invert Depth | off | — | Swap near/far brightness |
| Normal Space | camera | camera, world | Coordinate space for normals |
| Effect Opacity | 1.0 | 0.0-1.0 | Blend with original frame |
| Temporal Smoothing | 0.3 | 0.0-0.95 | EMA smoothing between frames |

## Performance

MoGe runs as a single forward pass (no iterative diffusion), so throughput scales directly with resolution and GPU.

On an RTX 4090 at 512px inference resolution:

- ~7-9GB VRAM
- Temporal smoothing masks any per-frame latency variation

Reduce inference resolution to 256 for faster throughput at the cost of geometric detail.

## How it works

1. Video frame arrives from Scope as a tensor
2. Frame is resized to the configured inference resolution (preserving aspect ratio)
3. A full-frame mask is generated (the entire scene is analyzed)
4. SAM 3D's MoGe model produces a 3D pointmap: a `(3, H, W)` tensor where each pixel has an (X, Y, Z) coordinate in camera space. This is a single forward pass through the network.
5. **Depth**: the Z channel of the pointmap is extracted, normalized to [0, 1], and mapped through a colormap LUT
6. **Normals**: horizontal and vertical gradients of the XYZ pointmap are computed, then cross-multiplied to produce per-pixel surface normal vectors. These are encoded as RGB where each channel maps from [-1, 1] to [0, 1].
7. Output is resized back to the original video resolution and optionally blended with the input frame

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings. No reinstall needed.

```bash
git clone https://github.com/hthillman/scope-sam3d.git
cd scope-sam3d
pip install -e .
```

## References

- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Meta's 3D reconstruction model
- [SAM 3D paper](https://arxiv.org/abs/2511.16624)
- [MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision](https://arxiv.org/abs/2410.19115)
- [Scope plugin docs](https://docs.daydream.live/scope/guides/plugins)
