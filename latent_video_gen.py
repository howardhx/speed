"""Training-free Spectral Progressive Diffusion inference for latent video generation using WAN 2.1.

Runs WAN at progressive spatial scales while keeping the temporal dimension
unchanged.

Usage
-----
    python latent_video_gen.py --prompts "a dog running in a meadow" \\
                               --transform dct --scales 0.5 1.0 \\
                               --save_dir ./out
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm.auto import tqdm


from utils import (
    delta_optimal_transitions,
    find_first_step_below,
    load_config,
    load_prompts,
    parse_scales,
    reset_scheduler_state,
    spectral_expand_and_align,
)

LOG = logging.getLogger("speed.wan21")


# =============================================================================
# Model loading (native WAN repo)
# =============================================================================

def load_wan(checkpoint_dir: str, repo_path: str, device: str):
    """Load WAN 2.1 T2V and return ``(pipeline, model, cfg)``.

    Args:
    - checkpoint_dir: Path to the WAN 2.1 checkpoint directory.
    - repo_path: Path to the WAN source repository (prepended to ``sys.path``).
    - device: Target GPU device, e.g. ``'cuda'`` or ``'cuda:0'``.

    Returns:
    - A ``(pipeline, model, cfg)`` tuple.
    """
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import wan  # type: ignore
    from wan.configs import wan_t2v_1_3B  # type: ignore

    cfg = wan_t2v_1_3B.t2v_1_3B
    # Accept either "cuda" (current device) or "cuda:N" (explicit index).
    device_id = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    pipeline = wan.WanT2V(config=cfg, checkpoint_dir=checkpoint_dir, device_id=device_id, t5_cpu=True)
    model = pipeline.model
    model.eval().requires_grad_(False)
    return pipeline, model, cfg


def setup_scheduler(n_steps: int, shift: float, device: str):
    """Configure WAN's shifted flow-matching sigma schedule.

    Args:
    - n_steps: Number of denoising steps.
    - shift: Flow-matching shift applied to the linear sigma ramp.
    - device: Device the timesteps are placed on.

    Returns:
    - The configured ``FlowMatchEulerDiscreteScheduler``.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    raw = np.linspace(1.0, 0.0, n_steps + 1)[:-1]
    shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(device=device, sigmas=shifted.tolist())
    return scheduler


def encode_prompt(pipeline, prompt: str, device: str):
    """Encode conditional and unconditional WAN text contexts.

    Args:
    - pipeline: The loaded WAN pipeline providing the text encoder.
    - prompt: Text prompt to encode (conditional context).
    - device: Device the contexts are placed on.

    Returns:
    - A ``(ctx_cond, ctx_uncond)`` tuple of text contexts.
    """
    ctx_cond = [c.to(device) for c in pipeline.text_encoder([prompt], torch.device("cpu"))]
    neg_prompt = getattr(pipeline, "sample_neg_prompt", "")
    ctx_uncond = [c.to(device) for c in pipeline.text_encoder([neg_prompt], torch.device("cpu"))]
    return ctx_cond, ctx_uncond


def _seq_len(shape) -> int:
    """Return WAN transformer sequence length for a latent shape.

    Args:
    - shape: Latent shape whose trailing ``(T, H, W)`` axes set the length.

    Returns:
    - The transformer sequence length.
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    return math.ceil((H * W) / 4 * T)


def cfg_velocity(model, x: torch.Tensor, t: torch.Tensor, ctx_cond, ctx_uncond, guidance: float) -> torch.Tensor:
    """Run WAN dual-pass CFG and return the generated flow.

    Args:
    - model: WAN transformer model.
    - x: Current latent ``[C, T, H, W]``.
    - t: Current timestep tensor.
    - ctx_cond: Conditional text context.
    - ctx_uncond: Unconditional text context.
    - guidance: Classifier-free guidance scale.

    Returns:
    - The guided flow ``[C, T, H, W]``.
    """
    H, W = x.shape[-2], x.shape[-1]
    pad_h, pad_w = H % 2, W % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    seq_len = _seq_len(x.shape)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        v_c = model([x], t=t, context=ctx_cond, seq_len=seq_len)[0]
        v_u = model([x], t=t, context=ctx_uncond, seq_len=seq_len)[0]
    v = v_u + guidance * (v_c - v_u)
    if pad_h or pad_w:
        v = v[..., :H, :W]
    return v


def decode_and_save(pipeline, cfg, latent: torch.Tensor, path: str) -> None:
    """Decode a video latent through the VAE and save as MP4.

    Args:
    - pipeline: The loaded WAN pipeline providing the VAE.
    - cfg: WAN config (supplies ``param_dtype`` for the decode autocast).
    - latent: Video latent ``[C, T, H, W]`` to decode.
    - path: Output MP4 path.

    Returns:
    - None; the decoded video is written to ``path``.
    """
    with torch.cuda.amp.autocast(dtype=cfg.param_dtype), torch.no_grad():
        video = pipeline.vae.decode(latent.unsqueeze(0))[0].float()
    frames = ((video.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
    imageio.mimwrite(path, frames, fps=16, codec="libx264")


# =============================================================================
# Progressive generation loop
# =============================================================================

def generate(
    model, scheduler,
    ctx_cond, ctx_uncond,
    *,
    scales, transition_times,
    transform: str,
    H_lat: int, W_lat: int, T_lat: int, C: int,
    n_steps: int, guidance: float,
    seed: int, device: str,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """Run staged denoising and return the final latent.

    Args:
    - model: WAN transformer model.
    - scheduler: Flow-matching scheduler from ``setup_scheduler``.
    - ctx_cond: Conditional text context.
    - ctx_uncond: Unconditional text context.
    - scales: Strictly increasing resolution scale list ending at 1.0.
    - transition_times: Per-transition flow-matching times (length S-1).
    - transform: Spectral basis, one of ``'dct'``, ``'dwt'``, ``'fft'``.
    - H_lat: Full-resolution latent height.
    - W_lat: Full-resolution latent width.
    - T_lat: Latent temporal length (invariant across stages).
    - C: Number of latent channels.
    - n_steps: Number of denoising steps.
    - guidance: Classifier-free guidance scale.
    - seed: Base seed for noise and per-transition expansion.
    - device: Device the denoising runs on.
    - progress_desc: If set, show a tqdm bar over the denoising steps with this
      label; left ``None`` adds no bar.

    Returns:
    - The final full-resolution video latent ``[C, T_lat, H_lat, W_lat]``.
    """
    sigmas = scheduler.sigmas
    transition_steps = [find_first_step_below(sigmas, thr) for thr in transition_times]

    s0 = scales[0]
    h0, w0 = round(s0 * H_lat), round(s0 * W_lat)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((C, T_lat, h0, w0), generator=rng, dtype=torch.bfloat16).to(device)

    stage_starts = [0] + transition_steps
    stage_ends = transition_steps + [n_steps]

    bar = None
    if progress_desc is not None:
        bar = tqdm(total=n_steps, desc=progress_desc, leave=False)

    for stage, (start, end, s_stage) in enumerate(zip(stage_starts, stage_ends, scales)):
        h, w = round(s_stage * H_lat), round(s_stage * W_lat)
        LOG.info("stage %d:  %dx%dx%d  steps [%d, %d)", stage + 1, T_lat, h, w, start, end)

        for i in range(start, end):
            t = scheduler.timesteps[i]
            v = cfg_velocity(model, x, torch.stack([t]), ctx_cond, ctx_uncond, guidance)
            x = scheduler.step(v.unsqueeze(0), t, x.unsqueeze(0), return_dict=False)[0].squeeze(0)
            if bar is not None:
                bar.update(1)

        if stage + 1 < len(scales):
            s_next = scales[stage + 1]
            t_at_transition = sigmas[end].item() if end < n_steps else 0.0
            x, t_tilde = spectral_expand_and_align(
                x, target_hw=(round(s_next * H_lat), round(s_next * W_lat)),
                r=s_next / s_stage, t=t_at_transition, transform=transform,
                seed=seed + (stage + 1) * 10000,
            )

            scheduler.sigmas[end] = t_tilde
            scheduler.timesteps[end] = t_tilde * 1000.0
            reset_scheduler_state(scheduler, end)
            x = x.to(device=device, dtype=torch.bfloat16)
            LOG.info("  -> expand+align: t=%.4f r=%.4f  t_tilde=%.4f",
                     t_at_transition, s_next / s_stage, t_tilde)

    if bar is not None:
        bar.close()
    return x


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the WAN 2.1 generator.

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
                        "Decimals (0.5), fractions (2/3), or pixel heights "
                        "(480 720, last == --height). DCT/FFT accept any ratio; "
                        "DWT needs every consecutive ratio == 2.")
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=None)

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
    """CLI entry point: load config, run the generator, save MP4s.

    Args:
    - None (reads from ``sys.argv`` via ``parse_args``).

    Returns:
    - None; one MP4 per prompt is written under ``--save_dir``.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)s  %(message)s",
    )

    # Config loading
    cfg_node = load_config(args.config, "wan21")
    defaults = cfg_node["defaults"]
    n_steps = args.n_steps if args.n_steps is not None else defaults["n_steps"]
    guidance = args.guidance if args.guidance is not None else defaults["guidance"]
    shift = args.shift if args.shift is not None else defaults["shift"]
    height = args.height if args.height is not None else defaults["height"]
    width = args.width if args.width is not None else defaults["width"]
    num_frames = args.num_frames if args.num_frames is not None else defaults["num_frames"]

    scales = parse_scales(args.scales, height)

    pipeline, model, cfg = load_wan(
        cfg_node["checkpoint_dir"], cfg_node["repo_path"], args.device,
    )
    H_lat = height // cfg.vae_stride[1]
    W_lat = width // cfg.vae_stride[2]
    T_lat = (num_frames - 1) // cfg.vae_stride[0] + 1
    C = cfg_node["latent_channels"]

    A = cfg_node["power_spectrum"]["A"]
    beta = cfg_node["power_spectrum"]["beta"]
    transition_times = delta_optimal_transitions(
        scales, args.delta, A, beta, H_lat, W_lat,
    )
    LOG.info("scales=%s  transitions t*=%s  latent=%dx%dx%d",
             scales, transition_times, T_lat, H_lat, W_lat)

    os.makedirs(args.save_dir, exist_ok=True)
    prompts = load_prompts(args)

    # Generation
    for p_idx, prompt in enumerate(prompts):
        ctx_cond, ctx_uncond = encode_prompt(pipeline, prompt, args.device)
        scheduler = setup_scheduler(n_steps, shift, args.device)
        latent = generate(
            model, scheduler,
            ctx_cond, ctx_uncond,
            scales=scales, transition_times=transition_times,
            transform=args.transform,
            H_lat=H_lat, W_lat=W_lat, T_lat=T_lat, C=C,
            n_steps=n_steps, guidance=guidance,
            seed=args.seed + p_idx, device=args.device,
            progress_desc="denoising" if args.progress else None,
        )
        out_path = os.path.join(args.save_dir, f"p{p_idx:04d}.mp4")

        # Save
        decode_and_save(pipeline, cfg, latent, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
