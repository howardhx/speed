---
name: speed-generator
description: >-
  Generate images and videos with SPEED (Spectral Progressive Diffusion), a
  training-free method that runs diffusion at progressively increasing
  resolution for a large speedup over full-resolution sampling. Use this skill whenever the user
  wants to generate an image with FLUX.1-dev, a pixel-space image with PixelGen,
  or a video with WAN 2.1 — and especially when they mention SPEED,
  progressive resolution, faster/efficient/accelerated diffusion sampling,
  spectral noise expansion, timestep alignment, DCT/DWT/FFT resolution
  transitions, or running latent_image_gen / pixel_image_gen / latent_video_gen.
  Reach for it for any "generate an image/video efficiently" request here even
  if the user never says the word "SPEED".
---

# SPEED Generator

Spectral Progressive Diffusion (SPEED) generates an image or video by running
the denoising trajectory at **increasing spatial resolution** instead of full
resolution the whole way. At each transition
the latent is **spectrally expanded** (DCT/DWT/FFT) with noise-filled high
frequencies and the flow-matching timestep is **aligned**, so the model
continues seamlessly at the larger resolution. 

This skill drives three standalone CLI generators bundled in `scripts/`:

| Script | Model | Output | Space |
|---|---|---|---|
| `latent_image_gen.py` | FLUX.1-dev | PNG image | latent image |
| `pixel_image_gen.py`  | PixelGen   | PNG image | pixel-space |
| `latent_video_gen.py` | WAN 2.1    | MP4 video | latent video |

`utils.py` holds the shared spectral math; `configs.yaml` holds per-model
checkpoint paths, power-spectrum fits, and defaults.

## When to use which generator

- **FLUX.1-dev** (`latent_image_gen.py`) — the default for a normal text-to-image
  request. Fast, high quality, guidance-distilled.
- **PixelGen** (`pixel_image_gen.py`) — pixel-space generation (square only). Use
  only when the user specifically wants PixelGen.
- **WAN 2.1** (`latent_video_gen.py`) — text-to-video.

If the user just says "generate an image of X efficiently" with no model named,
use **FLUX.1-dev**.

## Setup (do this once per session before generating)

1. **Environment**: The scripts do
   `from utils import ...`, so the bundled `scripts/` directory must be on
   `PYTHONPATH`. Use the correct environment with all dependencies from the speed/ repo installed.

2. **Checkpoints via env vars** (`configs.yaml` expands `${...}` placeholders;
   an unset one raises a clear `KeyError`). Set only what the chosen model needs:
   ```bash
   export FLUX_DIR=/local/howard/efficient_diffusion/wavelet_diffusion/models/FLUX.1-dev
   export PIXELGEN_REPO=/home/howard/efficient_diffusion/PixelGen
   export PIXELGEN_CKPT=/local/howard/efficient_diffusion/models/pixelgen/PixelGen_XXL_T2I.ckpt
   export PIXELGEN_CONFIG=$PIXELGEN_REPO/configs_t2i/sft_res512.yaml
   export WAN_PATH=/home/howard/efficient_diffusion/Wan2.1
   export WAN_CKPT=/local/howard/efficient_diffusion/models/Wan2.1-T2V-1.3B
   ```

## Running a generation

Always pass `PYTHONPATH=<skill>/scripts` and `CUDA_VISIBLE_DEVICES=$GPU`. Use
`--device cuda:0` (it maps to the pinned GPU) and `--progress` for a denoising
progress bar.

**FLUX image (the common case):**
```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=scripts \
python scripts/latent_image_gen.py \
    --prompts "a translucent jellyfish glowing in deep water" \
    --transform dct --scales 0.5 1.0 --delta 0.01 \
    --save_dir ./out_flux --device cuda:0 --progress
```

**PixelGen image:**
```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=scripts \
python scripts/pixel_image_gen.py \
    --prompts "a cute puppy" --transform dct --scales 0.5 1.0 \
    --save_dir ./out_pixelgen --device cuda:0 --progress
```

**WAN video (480p → 720p):**
```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=scripts \
python scripts/latent_video_gen.py \
    --prompts "a dog running in a meadow" \
    --transform dct --scales 480 720 \
    --save_dir ./out_wan --device cuda:0 --progress
```

Each script prints the output path(s) it wrote (one `pNNNN.png` / `.mp4` per
prompt).

## Shared flags

| Flag | Meaning |
|---|---|
| `--prompts "a" "b"` / `--prompt_txt f.txt` / `--prompt_csv f.csv` | Prompt input (mutually exclusive; CSV column `prompt`). |
| `--transform {dct,dwt,fft}` | Spectral basis at each transition. Default `dct`. |
| `--scales ...` | Increasing stage sizes ending at full resolution. See below. |
| `--delta` | Noise-dominated tolerance for transition scheduling. Default `0.01`. Smaller = transition later. |
| `--n_steps`, `--guidance` | Override the per-model defaults in `configs.yaml`. |
| `--height`, `--width` | Output size (defaults per model: FLUX 1024², PixelGen 512², WAN 720×1280). |
| `--seed`, `--save_dir`, `--device`, `--verbose`, `--progress` | Reproducibility, I/O, logging, progress bar. |
| WAN extra: `--num_frames`, `--shift` · PixelGen extra: `--timeshift`, `--neg_prompt` | Model-specific. |

### `--scales` — three accepted forms (freely mixed)

The stage list must be strictly increasing and end at full resolution. Each
value may be:
- a **decimal scale** — `0.5 1.0`, `0.37 1.0`
- a **fraction** — `1/2 1`, `2/3 1`, `1/3 2/3 1`
- a **pixel height** — `480 720` (the last value must equal `--height`)

`--scales 1.0` runs the plain full-resolution baseline (no progressive stages).

**Transform vs. scales:** DCT and FFT accept *any* ratio between consecutive
scales (so `0.37 1.0` is fine). **DWT is restricted to a 2× ratio** between every
consecutive pair (e.g. `0.25 0.5 1.0`) because Haar IDWT is structurally a 2×
upsample — it raises a clear error otherwise. When in doubt, use `dct`.

## Choosing settings

- **Default schedule**: `--transform dct --scales 0.5 1.0 --delta 0.01`. This is
  the canonical two-stage config and a good starting point.
- **More speedup**: add an earlier stage, e.g. `--scales 0.25 0.5 1.0` (DWT/DCT)
  or `--scales 0.33 0.67 1.0` (DCT/FFT). Lower scales = cheaper early steps but
  more aggressive; very low first scales can degrade quality.
- **Fewer steps** = faster but lower quality. `--n_steps` overrides the model
  default (FLUX/WAN 50, PixelGen 25).