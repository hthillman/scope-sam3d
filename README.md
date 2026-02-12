# scope-sam3d

Depth and normal maps from Meta's SAM 3D Objects geometry model, built as a [Daydream Scope](https://github.com/daydreamlive/scope) plugin.

Takes a video frame, runs it through the SAM 3D geometry pipeline (1.2B parameter flow transformer), and renders the resulting 3D geometry as a depth map, surface normal map, or both. No 3D mesh export or Gaussian splat viewer needed; the 3D understanding is projected back into 2D video frames.

## Prerequisites

- NVIDIA GPU with 12GB+ VRAM (tested on RTX 4090, A100)
- Accept the [SAM License](https://huggingface.co/facebook/sam-3d-objects) on HuggingFace
- Set `HF_TOKEN` environment variable or run `huggingface-cli login`

## Installation

In Scope, go to **Settings > Plugins** and enter one of:

**Git URL:**
```
git+https://github.com/daydreamlive/scope-sam3d.git
```

**Local path (development):**
```
/path/to/scope-sam3d
```

First load will download ~5GB of model weights from HuggingFace.

## Parameters

| Parameter | Type | Default | Range | Load-time | Description |
|-----------|------|---------|-------|-----------|-------------|
| Output | dropdown | depth | depth, normals, both | No | What to render |
| Depth Colormap | dropdown | magma | magma, viridis, plasma, grayscale | No | Colormap for depth visualization |
| Invert Depth | toggle | off | — | No | Swap near/far brightness |
| Normal Space | dropdown | camera | camera, world | No | Coordinate space for normals |
| Effect Opacity | slider | 1.0 | 0.0–1.0 | No | Blend with original frame |
| Inference Resolution | slider | 512 | 256–1024 | Yes | Resolution for model inference |
| Temporal Smoothing | slider | 0.3 | 0.0–0.95 | No | EMA smoothing between frames |
| Diffusion Steps | slider | 12 | 1–50 | Yes | Quality vs. speed tradeoff |

## Performance

On an RTX 4090 at 512px resolution with 12 diffusion steps:

- ~200–500ms per frame (2–5 FPS)
- ~7–9GB VRAM
- Temporal smoothing masks latency for a smoother feel

Reduce diffusion steps to 6–8 or inference resolution to 256 for faster throughput at the cost of detail.

## How it works

1. Video frame arrives from Scope as a tensor
2. A full-frame mask is generated (depth for the entire scene)
3. SAM 3D's sparse structure generator produces a coarse voxel grid
4. The dense geometry generator creates Gaussian splat representations
5. Gaussians are projected back to 2D via orthographic projection
6. Depth: z-values are colormapped. Normals: surface orientations are estimated from the point cloud via local PCA and encoded as RGB.

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings. No reinstall needed.

```bash
# Clone and install in development mode
git clone https://github.com/daydreamlive/scope-sam3d.git
cd scope-sam3d
pip install -e .
```

## References

- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Meta's 3D reconstruction model
- [SAM 3D paper](https://arxiv.org/abs/2511.16624)
- [Scope plugin docs](https://docs.daydream.live/scope/guides/plugins)
