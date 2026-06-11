"""Training-free Spectral Progressive Diffusion inference for pixel-space image generation using PixelGen.

Runs PixelGen directly in pixel space with progressive spatial resolution scales. 

Usage
-----
    python pixel_image_gen.py --prompts "a cute puppy" \\
                              --transform dct --scales 0.5 1.0 \\
                              --save_dir ./out
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from utils import (
    delta_optimal_transitions,
    load_config,
    load_prompts,
    parse_scales,
    spectral_expand_and_align,
)

LOG = logging.getLogger("speed.pixelgen")


# =============================================================================
# PixelGen-side imports (PIXELGEN_REPO must be on PYTHONPATH or env-set)
# =============================================================================

def _attach_pixelgen_repo() -> None:
    """Ensure the PixelGen source repository is importable.

    Args:
    - None (reads the ``PIXELGEN_REPO`` environment variable).

    Returns:
    - None; prepends ``PIXELGEN_REPO`` to ``sys.path`` if not already present.
    """
    repo = os.environ.get("PIXELGEN_REPO")
    if repo is None:
        raise RuntimeError(
            "PIXELGEN_REPO is not set; clone the PixelGen repository and set "
            "PIXELGEN_REPO to its root."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _instantiate(cfg_node) -> object:
    """Construct an object from an OmegaConf config node.

    Args:
    - cfg_node: Config node with ``class_path`` and optional ``init_args``.

    Returns:
    - The constructed object.
    """
    init_args = dict(cfg_node.get("init_args", {}) or {})
    module_name, class_name = cfg_node["class_path"].rsplit(".", 1)
    cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    return cls(**init_args)


def _load_ema_into(denoiser, ckpt: dict) -> None:
    """Load EMA denoiser weights from a PixelGen checkpoint.

    Args:
    - denoiser: Denoiser module to copy weights into (in place).
    - ckpt: Loaded checkpoint dict containing ``state_dict``.

    Returns:
    - None; EMA weights are copied into ``denoiser`` in place.
    """
    prefix = "ema_denoiser."
    sd = ckpt["state_dict"]
    for k, v in denoiser.state_dict().items():
        src = sd.get(prefix + k)
        if src is not None:
            v.copy_(src)


# =============================================================================
# Position-Interpolation patches  (sincos pos_embed + RoPE)
# =============================================================================

_pos_embed_cache: dict = {}


def _pi_sincos_pos_embed(
    hidden_size: int, grid_size: int, trained_grid: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Return a 2D sin-cos embedding sampled by position interpolation.

    Args:
    - hidden_size: Embedding dimensionality.
    - grid_size: Target grid size (patches per side) to sample.
    - trained_grid: Grid size the model was trained at (interpolation source).
    - device: Device the embedding is placed on.
    - dtype: Dtype of the returned embedding.

    Returns:
    - The sin-cos position embedding ``[1, grid_size**2, hidden_size]``.
    """
    key = (hidden_size, grid_size, trained_grid)
    if key in _pos_embed_cache:
        return _pos_embed_cache[key].to(device=device, dtype=dtype)
    pos = np.linspace(0.0, float(trained_grid - 1), grid_size, dtype=np.float32)
    half = hidden_size // 2
    omega = np.arange(half // 2, dtype=np.float64) / (half / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    grid_h, grid_w = np.meshgrid(pos, pos)
    pos_h, pos_w = grid_h.reshape(-1), grid_w.reshape(-1)
    eh = np.einsum("m,d->md", pos_h, omega)
    ew = np.einsum("m,d->md", pos_w, omega)
    emb_h = np.concatenate([np.sin(eh), np.cos(eh)], axis=1)
    emb_w = np.concatenate([np.sin(ew), np.cos(ew)], axis=1)
    pe = np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)
    pe = torch.from_numpy(pe).float().unsqueeze(0)
    _pos_embed_cache[key] = pe
    return pe.to(device=device, dtype=dtype)


def _set_pos_embed(denoiser, res: int, trained_grid: int) -> None:
    """Set PixelGen position embeddings for ``res``.

    Args:
    - denoiser: Denoiser whose ``pos_embed`` and patch scalings are updated.
    - res: Spatial resolution (pixels) for the current stage.
    - trained_grid: Grid size the model was trained at (interpolation source).

    Returns:
    - None; the denoiser's ``pos_embed`` and patch scalings are updated in place.
    """
    grid_size = res // denoiser.patch_size
    if grid_size ** 2 != denoiser.pos_embed.shape[1]:
        pe = _pi_sincos_pos_embed(
            denoiser.hidden_size, grid_size, trained_grid,
            denoiser.pos_embed.device, denoiser.pos_embed.dtype,
        )
        denoiser.pos_embed = torch.nn.Parameter(pe, requires_grad=False)
        denoiser.num_patches = grid_size ** 2
    denoiser.decoder_patch_scaling_h = res / 512.0
    denoiser.decoder_patch_scaling_w = res / 512.0


def _pi_fetch_pos_factory(trained_grid: int):
    """Build a ``denoiser.fetch_pos`` method using position interpolation.

    Args:
    - trained_grid: Grid size the model was trained at (interpolation source).

    Returns:
    - A ``fetch_pos(self, height, width, device)`` method to bind to the denoiser.
    """
    from src.models.layers.rope import precompute_freqs_cis_ex2d  # type: ignore

    def fetch_pos(self, height: int, width: int, device):
        """Return interpolated RoPE frequencies for ``(height, width)``.

        Args:
        - self: The denoiser this method is bound to.
        - height: Latent grid height (patches) for the current stage.
        - width: Latent grid width (patches) for the current stage.
        - device: Device the frequencies are placed on.

        Returns:
        - The (cached) interpolated RoPE frequency tensor for ``(height, width)``.
        """
        key = (height, width)
        if key in self.precompute_pos:
            return self.precompute_pos[key].to(device)
        head_dim = self.hidden_size // self.num_groups
        scale = (trained_grid / float(height), trained_grid / float(width))
        pos = precompute_freqs_cis_ex2d(
            head_dim, height, width, theta=10000.0, scale=scale,
        ).to(device)
        self.precompute_pos[key] = pos
        return pos

    return fetch_pos


# =============================================================================
# Model loading
# =============================================================================

def load_pixelgen(ckpt_path: str, config_path: str, device: str) -> Tuple:
    """Load PixelGen and return ``(vae, denoiser, conditioner, trained_grid)``.

    Args:
    - ckpt_path: Path to the PixelGen checkpoint (EMA weights are loaded).
    - config_path: Path to the PixelGen OmegaConf model config.
    - device: Target device for the loaded modules.

    Returns:
    - A ``(vae, denoiser, conditioner, trained_grid)`` tuple.
    """
    _attach_pixelgen_repo()

    cfg = OmegaConf.load(config_path)
    vae = _instantiate(cfg.model.vae)
    denoiser = _instantiate(cfg.model.denoiser)
    conditioner = _instantiate(cfg.model.conditioner)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_ema_into(denoiser, ckpt)
    del ckpt

    vae = vae.to(device).eval()
    denoiser = denoiser.to(device).eval()
    conditioner = conditioner.to(device).eval()

    trained_grid = 512 // denoiser.patch_size
    denoiser.fetch_pos = types.MethodType(_pi_fetch_pos_factory(trained_grid), denoiser)
    denoiser.precompute_pos = {}
    return vae, denoiser, conditioner, trained_grid


# =============================================================================
# Manual Euler sampler  (PixelGen's native schedule)
# =============================================================================

def make_schedule(n_steps: int, shift: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return PixelGen's shifted Euler schedule.

    Args:
    - n_steps: Number of denoising steps.
    - shift: Timeshift applied to the linear schedule.

    Returns:
    - A ``(ts, dts)`` tuple: the timestep grid and its consecutive differences.
    """
    ts = torch.linspace(0.0, 1.0 - 1.0 / n_steps, n_steps)
    ts = torch.cat([ts, torch.tensor([1.0])], dim=0)
    ts = ts / (ts + (1.0 - ts) * shift)
    return ts, ts[1:] - ts[:-1]


def cfg_velocity(
    denoiser, x: torch.Tensor, t_cur: torch.Tensor,
    cond, uncond, guidance: float,
) -> torch.Tensor:
    """Run classifier-free guidance and return the flow velocity.

    Args:
    - denoiser: PixelGen denoiser module.
    - x: Current pixel-space state ``[1, 3, res, res]``.
    - t_cur: Current flow-matching time tensor.
    - cond: Conditional text embedding.
    - uncond: Unconditional text embedding.
    - guidance: Classifier-free guidance scale.

    Returns:
    - The guided flow velocity.
    """
    cfg_x = torch.cat([x, x], dim=0)
    cfg_t = t_cur.repeat(2)
    cfg_cond = torch.cat([uncond, cond], dim=0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x0 = denoiser(cfg_x, cfg_t, cfg_cond)
    v = (x0 - cfg_x) / (1.0 - cfg_t.view(-1, 1, 1, 1)).clamp_min(5e-2)
    v_uncond, v_cond = v.chunk(2)
    return v_uncond + guidance * (v_cond - v_uncond)


# =============================================================================
# Progressive generation loop
# =============================================================================

def generate(
    denoiser, cond, uncond,
    *,
    scales, transition_times,
    transform: str,
    height: int, width: int,
    n_steps: int, guidance: float, timeshift: float,
    seed: int, device: str, trained_grid: int,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """Run the staged denoising loop in pixel space and return the final tensor.

    Args:
    - denoiser: PixelGen denoiser module.
    - cond: Conditional text embedding.
    - uncond: Unconditional text embedding.
    - scales: Strictly increasing resolution scale list ending at 1.0.
    - transition_times: Per-transition flow-matching times (length S-1).
    - transform: Spectral basis, one of ``'dct'``, ``'dwt'``, ``'fft'``.
    - height: Full image height in pixels (must equal width).
    - width: Full image width in pixels (must equal height).
    - n_steps: Number of denoising steps.
    - guidance: Classifier-free guidance scale.
    - timeshift: Timeshift for the Euler schedule.
    - seed: Base seed for noise and per-transition expansion.
    - device: Device the denoising runs on.
    - trained_grid: Grid size the denoiser was trained at.
    - progress_desc: If set, show a tqdm bar over the denoising steps with this
      label; left ``None`` adds no bar.

    Returns:
    - The final full-resolution pixel-space tensor ``[1, 3, res, res]``.
    """
    if height != width:
        raise ValueError("PixelGen denoiser is square-only; height must equal width.")
    res_full = height

    ts, dts = make_schedule(n_steps, timeshift)
    ts, dts = ts.to(device), dts.to(device)
    sigmas = 1.0 - ts  # flow-matching noise level (1 = pure noise)

    # Locate transition step indices on the sigma schedule.
    sigmas_list = sigmas.tolist()
    transition_steps: list[int] = []
    for thresh in transition_times:
        idx = next((i for i in range(n_steps) if sigmas_list[i] <= thresh), n_steps)
        transition_steps.append(idx)

    # Initial noise at the coarsest scale.
    s0 = scales[0]
    res0 = round(s0 * res_full)
    _set_pos_embed(denoiser, res0, trained_grid)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, 3, res0, res0), generator=g).to(device)
    t_cur = torch.zeros([1], device=device)

    stage_starts = [0] + transition_steps
    stage_ends = transition_steps + [n_steps]

    bar = None
    if progress_desc is not None:
        bar = tqdm(total=n_steps, desc=progress_desc, leave=False)

    for stage, (start, end, s_stage) in enumerate(zip(stage_starts, stage_ends, scales)):
        res_stage = round(s_stage * res_full)
        LOG.info("stage %d:  %dx%d  steps [%d, %d)", stage + 1, res_stage, res_stage, start, end)
        for i in range(start, end):
            v = cfg_velocity(denoiser, x, t_cur, cond, uncond, guidance)
            x = x + v * dts[i]
            t_cur = t_cur + dts[i]
            if bar is not None:
                bar.update(1)

        if stage + 1 < len(scales):
            s_next = scales[stage + 1]
            t_at_transition = sigmas[end].item() if end < n_steps else 0.0
            res_next = round(s_next * res_full)
            x, t_tilde = spectral_expand_and_align(
                x, target_hw=(res_next, res_next), r=s_next / s_stage,
                t=t_at_transition, transform=transform,
                seed=seed + (stage + 1) * 10000,
            )

            # Patch the manual schedule at the transition step so the next
            # forward pass sees the aligned time t_tilde.
            t_eff_for_t = 1.0 - t_tilde
            ts[end] = t_eff_for_t
            if end < n_steps:
                dts[end - 1] = ts[end] - ts[end - 1]
                dts[end] = ts[end + 1] - ts[end]
            t_cur = torch.tensor([t_eff_for_t], device=device, dtype=t_cur.dtype)

            _set_pos_embed(denoiser, round(s_next * res_full), trained_grid)
            LOG.info("  -> expand+align: t=%.4f r=%.4f  t_tilde=%.4f",
                     t_at_transition, s_next / s_stage, t_tilde)

    if bar is not None:
        bar.close()
    return x


def decode_and_save(vae, samples: torch.Tensor, path: str) -> None:
    """Decode pixel-space samples through the VAE and save as PNG.

    Args:
    - vae: PixelGen autoencoder providing ``decode``.
    - samples: Pixel-space samples ``[1, 3, res, res]`` to decode.
    - path: Output PNG path.

    Returns:
    - None; the decoded image is written to ``path``.
    """
    from src.models.autoencoder.base import fp2uint8  # type: ignore
    with torch.no_grad():
        img = fp2uint8(vae.decode(samples))
    arr = img.permute(0, 2, 3, 1).cpu().numpy()[0]
    Image.fromarray(arr).save(path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PixelGen generator.

    Args:
    - None (reads from ``sys.argv``).

    Returns:
    - The parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompts", nargs="+")
    prompts.add_argument("--prompt_txt", type=str)
    prompts.add_argument("--prompt_csv", type=str)

    p.add_argument("--transform", choices=("dct", "dwt", "fft"), default="dct")
    p.add_argument("--scales", nargs="+", type=str, default=["0.5", "1.0"],
                   help="Stage sizes, increasing, ending at full resolution. "
                        "Decimals (0.5), fractions (1/2, 2/3), or pixel heights "
                        "(256 512, last == --height). DCT/FFT accept any ratio; "
                        "DWT needs every consecutive ratio == 2.")
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--timeshift", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--neg_prompt", type=str, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress", action="store_true",
                   help="Show a tqdm progress bar over the denoising steps.")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).with_name("configs.yaml")))
    return p.parse_args()


def main() -> None:
    """CLI entry point: load config, run the generator, save PNGs.

    Args:
    - None (reads from ``sys.argv`` via ``parse_args``).

    Returns:
    - None; one PNG per prompt is written under ``--save_dir``.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)s  %(message)s",
    )

    # Config loading
    cfg = load_config(args.config, "pixelgen")
    defaults = cfg["defaults"]
    n_steps = args.n_steps if args.n_steps is not None else defaults["n_steps"]
    guidance = args.guidance if args.guidance is not None else defaults["guidance"]
    timeshift = args.timeshift if args.timeshift is not None else defaults["timeshift"]
    height = args.height if args.height is not None else defaults["height"]
    width = args.width if args.width is not None else defaults["width"]
    neg_prompt = args.neg_prompt if args.neg_prompt is not None else defaults["neg_prompt"]

    scales = parse_scales(args.scales, height)
    A = cfg["power_spectrum"]["A"]
    beta = cfg["power_spectrum"]["beta"]
    transition_times = delta_optimal_transitions(
        scales, args.delta, A, beta, height, width,
    )
    LOG.info("scales=%s  transitions t*=%s", scales, transition_times)

    os.makedirs(args.save_dir, exist_ok=True)
    prompts = load_prompts(args)

    vae, denoiser, conditioner, trained_grid = load_pixelgen(
        cfg["checkpoint_path"], cfg["config_path"], args.device,
    )

    # Generation
    for p_idx, prompt in enumerate(prompts):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond, uncond = conditioner([prompt], {"negative_prompt": neg_prompt})
        samples = generate(
            denoiser, cond, uncond,
            scales=scales, transition_times=transition_times,
            transform=args.transform,
            height=height, width=width,
            n_steps=n_steps, guidance=guidance, timeshift=timeshift,
            seed=args.seed + p_idx, device=args.device,
            trained_grid=trained_grid,
            progress_desc="denoising" if args.progress else None,
        )
        out_path = os.path.join(args.save_dir, f"p{p_idx:04d}.png")

        # Save
        decode_and_save(vae, samples, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
