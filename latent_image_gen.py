"""Training-free Spectral Progressive Diffusion inference for latent image
generation using FLUX.1-dev.

Runs FLUX.1-dev at progressive spatial scales.

Usage
-----
    python latent_image_gen.py --prompts "a cute puppy" \\
                               --transform dct --scales 0.5 1.0 \\
                               --save_dir ./out
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
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

from diffusers import FluxPipeline

LOG = logging.getLogger("speed.flux")


# =============================================================================
# Pipeline loading and scheduler setup
# =============================================================================

def load_pipeline(checkpoint_dir: str, device: str):
    """Load FLUX.1-dev with bfloat16 weights.

    Args:
    - checkpoint_dir: Local path or HF repo passed to ``from_pretrained``.
    - device: Target GPU device, e.g. ``'cuda'`` or ``'cuda:0'``.

    Returns:
    - The loaded ``FluxPipeline``.
    """
    pipe = FluxPipeline.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    # Accept either "cuda" or "cuda:N" (explicit index).
    idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(idx)
    # >= 42GB GPU VRAM required to load the model fully on GPU; otherwise
    # offload to CPU.
    if free_bytes > 42 * (1024 ** 3):
        pipe.to(device)
    else:
        pipe.enable_model_cpu_offload(device=device)
    for module in (pipe.vae, pipe.transformer, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False)
    pipe.transformer.eval()
    return pipe


def setup_scheduler(pipe, n_steps: int, H_lat: int, W_lat: int, device: str):
    """Configure the FLUX flow-matching scheduler.

    Args:
    - pipe: The loaded ``FluxPipeline`` whose scheduler is configured.
    - n_steps: Number of denoising steps.
    - H_lat: Full-resolution latent height (sets the dynamic shift ``mu``).
    - W_lat: Full-resolution latent width (sets the dynamic shift ``mu``).
    - device: Device the timesteps are placed on.

    Returns:
    - The configured scheduler with timesteps and sigmas set.
    """
    scheduler = pipe.scheduler
    seq_len = (H_lat // 2) * (W_lat // 2)
    cfg = scheduler.config
    mu = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len) \
        * (seq_len - cfg.base_image_seq_len) + cfg.base_shift
    sigmas = np.linspace(1.0, 1.0 / n_steps, n_steps).tolist()
    scheduler.set_timesteps(device=device, sigmas=sigmas, mu=mu)
    return scheduler


# =============================================================================
# FLUX-specific helpers (packed-latent representation)
# =============================================================================

def encode_prompt(pipe, prompt: str, device: str):
    """Encode a text prompt into FLUX conditioning tensors.

    Args:
    - pipe: The loaded ``FluxPipeline`` providing the text encoders.
    - prompt: Text prompt to encode.
    - device: Device the embeddings are placed on.

    Returns:
    - A ``(prompt_embeds, pooled_embeds, text_ids)`` tuple.
    """
    with torch.no_grad():
        prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
    return prompt_embeds, pooled_embeds, text_ids


def pack(x_spatial: torch.Tensor, C: int) -> torch.Tensor:
    """Pack spatial FLUX latents into sequence form.

    Args:
    - x_spatial: Spatial latent ``[B, C, H, W]``.
    - C: Number of latent channels.

    Returns:
    - The packed sequence-form latent.
    """
    H, W = x_spatial.shape[-2], x_spatial.shape[-1]
    return FluxPipeline._pack_latents(x_spatial, 1, C, H, W)


def unpack(x_packed: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Unpack FLUX sequence latents into spatial form.

    Args:
    - x_packed: Packed sequence latent to unpack.
    - H: Latent height (in latent pixels) to unpack to.
    - W: Latent width (in latent pixels) to unpack to.

    Returns:
    - The spatial-form latent ``[B, C, H, W]``.
    """
    return FluxPipeline._unpack_latents(x_packed, H * 8, W * 8, 8)


_img_id_cache: dict = {}


def make_img_ids(H: int, W: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Return cached FLUX latent image IDs.

    Args:
    - H: Latent height (in latent pixels).
    - W: Latent width (in latent pixels).
    - device: Device the IDs are placed on.
    - dtype: Dtype of the returned IDs.

    Returns:
    - The (cached) latent image-ID tensor for ``(H, W)``.
    """
    key = (H, W, device, dtype)
    if key not in _img_id_cache:
        _img_id_cache[key] = FluxPipeline._prepare_latent_image_ids(
            1, H // 2, W // 2, device, dtype
        )
    return _img_id_cache[key]


def decode_and_save(pipe, latent: torch.Tensor, path: str, device: str) -> None:
    """Decode a full-resolution latent through the VAE and save as PNG.

    Args:
    - pipe: The loaded ``FluxPipeline`` providing the VAE.
    - latent: Spatial latent ``[1, C, H, W]`` to decode.
    - path: Output PNG path.
    - device: Device used for the VAE decode.

    Returns:
    - None; the decoded image is written to ``path``.
    """
    from PIL import Image
    vae = pipe.vae
    scaling = float(vae.config.scaling_factor)
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    with torch.no_grad():
        z = latent.to(device=device, dtype=torch.bfloat16) / scaling + shift
        img = vae.decode(z, return_dict=False)[0].float().cpu().clamp(-1, 1)
    arr = ((img[0].permute(1, 2, 0) + 1) * 127.5).numpy().astype("uint8")
    Image.fromarray(arr).save(path)


# =============================================================================
# Progressive generation loop
# =============================================================================

def generate(
    pipe,
    scheduler,
    encoded_prompt: tuple,
    *,
    scales,
    transition_times,
    transform: str,
    H_lat: int,
    W_lat: int,
    C: int,
    n_steps: int,
    guidance: float,
    seed: int,
    device: str,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """Run the progressive resolution denoising loop and return the final
    full-resolution latent.

    Args:
    - pipe: The loaded ``FluxPipeline``.
    - scheduler: Flow-matching scheduler from ``setup_scheduler``.
    - encoded_prompt: ``(prompt_embeds, pooled_embeds, text_ids)`` tuple.
    - scales: Strictly increasing resolution scale list ending at 1.0.
    - transition_times: Per-transition flow-matching times (length S-1).
    - transform: Spectral basis, one of ``'dct'``, ``'dwt'``, ``'fft'``.
    - H_lat: Full-resolution latent height.
    - W_lat: Full-resolution latent width.
    - C: Number of latent channels.
    - n_steps: Number of denoising steps.
    - guidance: FLUX guidance value (guidance-distilled, single pass).
    - seed: Base seed for noise and per-transition expansion.
    - device: Device the denoising runs on.
    - progress_desc: If set, display a tqdm bar over the denoising steps with
      this label.

    Returns:
    - The final full-resolution spatial latent ``[1, C, H_lat, W_lat]``.
    """
    prompt_embeds, pooled_embeds, text_ids = encoded_prompt
    transformer = pipe.transformer
    guidance_t = torch.full([1], guidance, device=device, dtype=torch.float32).expand(1)

    # Locate scheduler step indices for each transition.
    sigmas = scheduler.sigmas
    transition_steps = [
        find_first_step_below(sigmas, threshold) for threshold in transition_times
    ]

    # FLUX packs 2x2 patches, so every latent grid must have even dimensions.
    def _stage_dims(s):
        """Return even ``(h, w)`` latent dims for scale ``s``.

        Args:
        - s: Scale (fraction of full resolution).

        Returns:
        - An even ``(h, w)`` latent-dimension tuple.
        """
        h = max(2, (round(s * H_lat) // 2) * 2)
        w = max(2, (round(s * W_lat) // 2) * 2)
        return h, w

    # Initial noise at the coarsest scale s_0.
    h0, w0 = _stage_dims(scales[0])
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x_spatial = torch.randn((1, C, h0, w0), generator=rng, dtype=torch.bfloat16).to(device)
    x_packed = pack(x_spatial, C)

    stage_starts = [0] + transition_steps
    stage_ends = transition_steps + [n_steps]

    bar = None
    if progress_desc is not None:
        bar = tqdm(total=n_steps, desc=progress_desc, leave=False)

    for stage, (start, end, s_stage) in enumerate(zip(stage_starts, stage_ends, scales)):
        h, w = _stage_dims(s_stage)
        img_ids = make_img_ids(h, w, device, torch.bfloat16)
        LOG.info("stage %d:  %dx%d  steps [%d, %d)", stage + 1, h, w, start, end)

        for i in range(start, end):
            t = scheduler.timesteps[i]
            t_norm = t.expand(x_packed.shape[0]).to(x_packed.dtype) / 1000.0
            v = transformer(
                hidden_states=x_packed,
                timestep=t_norm,
                guidance=guidance_t,
                pooled_projections=pooled_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            x_packed = scheduler.step(v, t, x_packed, return_dict=False)[0]
            if bar is not None:
                bar.update(1)

        if stage + 1 < len(scales):
            s_next = scales[stage + 1]
            t_at_transition = sigmas[end].item() if end < n_steps else 0.0
            x_spatial = unpack(x_packed, h, w)

            x_tilde, t_tilde = spectral_expand_and_align(
                x_spatial, target_hw=_stage_dims(s_next), r=s_next / s_stage,
                t=t_at_transition, transform=transform,
                seed=seed + (stage + 1) * 10000,
            )

            # Patch the scheduler at the transition step so the next forward
            # pass sees the aligned time t_tilde (paper Eq. 6) under the
            # original NFE budget.
            scheduler.sigmas[end] = t_tilde
            scheduler.timesteps[end] = t_tilde * 1000.0
            reset_scheduler_state(scheduler, end)

            x_packed = pack(x_tilde.to(device=device, dtype=torch.bfloat16), C)
            LOG.info("  -> expand+align: t=%.4f r=%.4f  t_tilde=%.4f",
                     t_at_transition, s_next / s_stage, t_tilde)

    if bar is not None:
        bar.close()
    return unpack(x_packed, H_lat, W_lat)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the progressive resolution latent image generator.

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
                        "(512 1024, last == --height). DCT/FFT accept any ratio; "
                        "DWT needs every consecutive ratio == 2.")
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--height", type=int, default=None, help="Image height in pixels.")
    p.add_argument("--width", type=int, default=None, help="Image width in pixels.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress", action="store_true",
                   help="Show a tqdm progress bar over the denoising steps.")
    p.add_argument(
        "--config", type=str,
        default=str(Path(__file__).with_name("configs.yaml")),
    )
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
    cfg = load_config(args.config, "flux")
    defaults = cfg["defaults"]
    n_steps = args.n_steps if args.n_steps is not None else defaults["n_steps"]
    guidance = args.guidance if args.guidance is not None else defaults["guidance"]
    height = args.height if args.height is not None else defaults["height"]
    width = args.width if args.width is not None else defaults["width"]
    C = cfg["latent_channels"]
    vae_scale = cfg["vae_downscale"]
    H_lat, W_lat = height // vae_scale, width // vae_scale

    scales = parse_scales(args.scales, height)
    A = cfg["power_spectrum"]["A"]
    beta = cfg["power_spectrum"]["beta"]
    transition_times = delta_optimal_transitions(
        scales, args.delta, A, beta, H_lat, W_lat,
    )
    LOG.info("scales=%s  transitions t*=%s", scales, transition_times)

    os.makedirs(args.save_dir, exist_ok=True)
    prompts = load_prompts(args)

    pipe = load_pipeline(cfg["checkpoint_dir"], args.device)

    # Generation
    for p_idx, prompt in enumerate(prompts):
        encoded = encode_prompt(pipe, prompt, args.device)
        scheduler = setup_scheduler(pipe, n_steps, H_lat, W_lat, args.device)
        latent = generate(
            pipe, scheduler, encoded,
            scales=scales, transition_times=transition_times,
            transform=args.transform, H_lat=H_lat, W_lat=W_lat,
            C=C, n_steps=n_steps, guidance=guidance,
            seed=args.seed + p_idx, device=args.device,
            progress_desc="denoising" if args.progress else None,
        )
        out_path = os.path.join(args.save_dir, f"p{p_idx:04d}.png")

        # Save
        decode_and_save(pipe, latent, out_path, args.device)
        print(out_path)


if __name__ == "__main__":
    main()
