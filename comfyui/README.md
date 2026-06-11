# Sampler SPEED for ComfyUI

This custom sampler runs Spectral Progressive Diffusion on top of ComfyUI samplers. The trajectory is segmented at each resolution
transition; between segments the latent is spectrally expanded and the
flow-matching time is aligned. Currently PixelGen is not supported. 

## Install
Firstly, please ensure that the required dependencies for ComfyUI are installed. Then, link the folder:
```bash
ln -s comfyui \
      /path/to/ComfyUI/custom_nodes/SPEED

# Install the node's extra dependency into the ComfyUI environment.
# (ComfyUI already ships torch, numpy, scipy, PyYAML and av; only PyWavelets
# is additional, used by the `dwt` transform.)
pip install -r /path/to/ComfyUI/custom_nodes/SPEED/requirements.txt
```

Restart ComfyUI. The node appears under
`sampling -> custom_sampling -> samplers` as **Sampler SPEED (Spectral
Progressive Diffusion)**.

This node runs inside ComfyUI's own Python environment (the one created from
ComfyUI's `requirements.txt`), not the `speed/` inference env.

## Inputs

| Field | Description |
|---|---|
| `base_sampler` | Underlying k-diffusion sampler. Multistep solvers restart at each transition because the schedule is segmented. |
| `transform` | Spectral basis at each transition: `dct` (default), `dwt`, or `fft`. DWT requires `s_{i+1}/s_i = 2` between consecutive scales. |
| `mode` | `delta_optimal` computes transitions from `scales`, `delta`, and the selected power-spectrum preset. `manual` uses user-specified sigma thresholds. |
| `model_preset` | `flux`, `wan21`, or `custom`. Picks `(A, beta)` from `speed/configs.yaml` for delta-optimal mode. |
| `scales` | Comma-separated resolution fractions ending at 1.0, for example `0.5,1.0` or `0.25,0.5,1.0`. |
| `delta` | Noise-dominated tolerance. Default `0.01`. Smaller values transition later. |
| `manual_sigmas` | Comma-separated sigma thresholds, one per transition. Used only in `manual` mode. |
| `spectrum_A`, `spectrum_beta` | Power-spectrum constants when `model_preset = custom`. |
| `seed` | Seed for spectral-noise padding at each transition. |

## Usage

Wire `Sampler SPEED` to `SamplerCustomAdvanced` like any other custom sampler.
ComfyUI's noise, scheduler, and model loaders feed in unchanged.

### FLUX.1-dev Example

`Sampler SPEED` drops into a standard FLUX.1-dev `SamplerCustomAdvanced` graph:

```text
UNETLoader (flux1-dev) ──┬─────────────────────► BasicGuider ─┐
DualCLIPLoader → CLIPTextEncode → FluxGuidance ──┘             │
RandomNoise ──────────────────────────────────────────────────┤
BasicScheduler (model, steps) ──────────► SIGMAS ─────────────┤──► SamplerCustomAdvanced
Sampler SPEED ──────────────────────────► SAMPLER ────────────┤        │
EmptySD3LatentImage ──────────────────────────────────────────┘        ▼
                                                          VAEDecode → SaveImage

Sampler SPEED settings:
  base_sampler = euler   transform = dct      mode = delta_optimal
  model_preset = flux    scales = 0.5,1.0     delta = 0.01
```

This matches the default two-stage FLUX configuration in
`speed/latent_image_gen.py` (`scales = 0.5,1.0`).

A ready-to-run, headless version of exactly this graph lives next to this file:
`workflow_flux_api.json` (API-format) plus `run_workflow.py`, which POSTs it to
a running ComfyUI server and waits for the image — no UI building required:

```bash
# with ComfyUI running on :8188
python run_workflow.py --server http://127.0.0.1:8188 \
                       --prompt "a corgi puppy sitting in a field" --seed 42
```

### WAN 2.1

WAN must be loaded through a ComfyUI wrapper that exposes a standard `MODEL`
output. Use `model_preset = wan21`; the remaining settings work the same way.

## Notes

- Wire a normal **full-resolution** latent + noise (`EmptySD3LatentImage` +
  `RandomNoise`), exactly like any other sampler — you do **not** build a
  low-res latent for stage 0. The node DCT-truncates that incoming full-res
  latent down to `scales[0]` (drop the high-frequency DCT coefficients, keep
  the low-frequency top-left block) to begin the coarsest stage. 
- For DWT, every consecutive `s_{i+1}/s_i` must equal 2. Use DCT or FFT for
  non-dyadic schedules.
- After a transition, the transition sigma is patched to the aligned time and
  sampling continues on the original schedule.
