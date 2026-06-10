"""ComfyUI sampler for Spectral Progressive Diffusion.

Wraps any ``comfy.k_diffusion.sampling.sample_*`` solver with the resolution
transitions from the paper. The denoising trajectory is segmented at each
transition; between segments the latent is spectrally expanded and
timestep-aligned.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch

import comfy.samplers
import comfy.utils
from comfy_api.latest import io
import comfy.k_diffusion.sampling as kds

# Import the sibling ``speed/utils.py`` by file path to avoid colliding with
# ComfyUI's own top-level ``utils`` package.
_SPEED_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_UTILS_PATH = os.path.join(_SPEED_ROOT, "utils.py")
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("speed_utils", _UTILS_PATH)
_speed_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_speed_utils)

_dct_expand_np = _speed_utils._dct_expand_np
_dwt_expand_np = _speed_utils._dwt_expand_np
_fft_expand_np = _speed_utils._fft_expand_np
align_timestep = _speed_utils.align_timestep
delta_optimal_transitions = _speed_utils.delta_optimal_transitions
kappa = _speed_utils.kappa
validate_scales = _speed_utils.validate_scales


# =============================================================================
# Per-model power-spectrum presets.
# =============================================================================

_PRESETS = {
    "flux":   {"A": 203.615097, "beta": 1.915461},
    "wan21":  {"A": 219.484718, "beta": 2.422687},
    "custom": None,
}


# =============================================================================
# Parsing helpers
# =============================================================================

def _parse_scales(s: str) -> List[float]:
    """Parse a comma-separated scale list."""
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    validate_scales(out)
    return out


def _parse_sigmas(s: str) -> List[float]:
    """Parse comma-separated manual transition sigmas."""
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if any(not (0.0 < v < 1.0) for v in out):
        raise ValueError(f"every manual sigma must be in (0, 1); got {out}")
    for a, b in zip(out[:-1], out[1:]):
        if not (a > b):
            raise ValueError(f"manual sigmas must be strictly decreasing; got {out}")
    return out


# =============================================================================
# Spectral transition helper.
# =============================================================================

def _expand_and_align_torch(
    x: torch.Tensor, s_i: float, s_next: float, t: float,
    transform: str, seed: int, H_full: int, W_full: int,
) -> Tuple[torch.Tensor, float]:
    """Expand a 4D image latent or 5D video latent over its spatial axes."""
    if transform not in ("dct", "dwt", "fft"):
        raise ValueError(f"transform must be dct|dwt|fft, got {transform!r}")
    r = s_next / s_i
    H_tgt = round(s_next * H_full)
    W_tgt = round(s_next * W_full)

    if x.ndim == 5:
        B, C, T, h_lo, w_lo = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, h_lo, w_lo)
    elif x.ndim == 4:
        x4 = x
    else:
        raise ValueError(f"expected 4D or 5D latent, got shape {tuple(x.shape)}")

    x_np = x4.detach().cpu().float().numpy()
    if transform == "dwt":
        if abs(r - 2.0) > 1e-6:
            raise ValueError(
                f"DWT requires r=2 between consecutive scales; got r={r:.4f}. "
                "Use transform=dct or transform=fft for non-dyadic ratios."
            )
        expanded = _dwt_expand_np(x_np, t, seed)
    elif transform == "dct":
        expanded = _dct_expand_np(x_np, (H_tgt, W_tgt), t, seed)
    else:
        expanded = _fft_expand_np(x_np, (H_tgt, W_tgt), t, seed)

    rescaled = (kappa(t, r) * expanded).astype(np.float32)
    x4_new = torch.from_numpy(rescaled).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        out = x4_new.reshape(B, T, C, H_tgt, W_tgt).permute(0, 2, 1, 3, 4)
    else:
        out = x4_new

    return out, align_timestep(t, r)


# =============================================================================
# Initial coarse-resolution latent.
# =============================================================================

def _initial_dct_downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale ``x`` by DCT truncation."""
    if scale >= 1.0:
        return x

    H_full, W_full = x.shape[-2], x.shape[-1]
    H_lo, W_lo = round(H_full * scale), round(W_full * scale)

    if x.ndim == 5:
        B, C, T, _, _ = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_full, W_full)
    else:
        x4 = x

    x_np = x4.detach().cpu().float().numpy()
    from scipy.fft import dctn, idctn
    out_np = np.empty(x_np.shape[:-2] + (H_lo, W_lo), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs = dctn(x_np[idx], type=2, norm="ortho")
        out_np[idx] = idctn(coeffs[:H_lo, :W_lo], type=2, norm="ortho").astype(np.float32)
    out4 = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        return out4.reshape(B, T, C, H_lo, W_lo).permute(0, 2, 1, 3, 4)
    return out4


# =============================================================================
# Transition scheduling.
# =============================================================================

def _resolve_transitions(
    sigmas: torch.Tensor, scales: List[float], delta: float, A: float, beta: float,
    H_full: int, W_full: int,
) -> List[Tuple[int, float, float]]:
    """Return ``(step_idx, s_i, s_next)`` transitions from ``scales`` and ``delta``."""
    if len(scales) < 2:
        return []
    t_stars = delta_optimal_transitions(scales, delta, A, beta, H_full, W_full)
    out: List[Tuple[int, float, float]] = []
    n_steps = len(sigmas) - 1
    for i, (s_old, s_new, t_thr) in enumerate(zip(scales[:-1], scales[1:], t_stars)):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= t_thr),
            n_steps,
        )
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


def _resolve_manual(
    sigmas: torch.Tensor, scales: List[float], manual_sigmas: List[float],
) -> List[Tuple[int, float, float]]:
    """Return transitions from user-specified sigma thresholds."""
    if len(scales) < 2:
        return []
    if len(manual_sigmas) != len(scales) - 1:
        raise ValueError(
            f"manual_sigmas has length {len(manual_sigmas)}, expected "
            f"{len(scales) - 1} (one threshold per transition in scales)."
        )
    out: List[Tuple[int, float, float]] = []
    n_steps = len(sigmas) - 1
    for s_old, s_new, thr in zip(scales[:-1], scales[1:], manual_sigmas):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= thr),
            n_steps,
        )
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


# =============================================================================
# Segmented sampling.
# =============================================================================

def _segment_callback(outer_cb, segment_start_idx: int):
    """Re-base callback step indices to the full schedule."""
    if outer_cb is None:
        return None
    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + segment_start_idx
        outer_cb(d)
    return inner


@torch.no_grad()
def sample_speed(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    *,
    transform: str = "dct",
    base_sampler: str = "euler",
    mode: str = "delta_optimal",
    scales: List[float] = None,
    delta: float = 0.01,
    spectrum_A: float = 203.615097,
    spectrum_beta: float = 1.915461,
    manual_sigmas: List[float] = None,
    seed: int = 0,
):
    """Comfy-compatible ``sample_*`` function."""
    extra_args = {} if extra_args is None else extra_args
    sampler_fn = getattr(kds, f"sample_{base_sampler}", None)
    if sampler_fn is None:
        raise ValueError(f"Unknown base sampler {base_sampler!r}.")

    H_full, W_full = x.shape[-2], x.shape[-1]

    if not scales or len(scales) < 2:
        return sampler_fn(model, x, sigmas, extra_args=extra_args,
                          callback=callback, disable=disable)
    first_scale = scales[0]
    if mode == "delta_optimal":
        transitions = _resolve_transitions(
            sigmas, scales, delta, spectrum_A, spectrum_beta, H_full, W_full,
        )
    elif mode == "manual":
        transitions = _resolve_manual(sigmas, scales, manual_sigmas or [])
    else:
        raise ValueError(f"mode must be delta_optimal|manual, got {mode!r}")

    # DCT-truncate the incoming latent down to the coarsest scale.
    if first_scale < 1.0:
        x = _initial_dct_downscale(x, first_scale)

    sigmas = sigmas.clone()
    segment_starts = [0] + [t[0] for t in transitions]

    for seg_i, seg_start in enumerate(segment_starts):
        seg_end = transitions[seg_i][0] if seg_i < len(transitions) else len(sigmas) - 1
        seg_sigmas = sigmas[seg_start:seg_end + 1]
        if len(seg_sigmas) >= 2:
            cb = _segment_callback(callback, seg_start)
            x = sampler_fn(model, x, seg_sigmas, extra_args=extra_args,
                           callback=cb, disable=disable)

        if seg_i >= len(transitions):
            break

        step_idx, s_i, s_next = transitions[seg_i]
        sigma_at_transition = float(sigmas[step_idx])
        x, t_tilde = _expand_and_align_torch(
            x, s_i, s_next, sigma_at_transition,
            transform=transform, seed=seed + (seg_i + 1) * 10000,
            H_full=H_full, W_full=W_full,
        )

        # Patch only the transition sigma, matching the reference inference loop.
        sigmas[step_idx] = float(t_tilde)

    return x


# =============================================================================
# Comfy node
# =============================================================================

def _list_samplers() -> List[str]:
    """Return supported k-diffusion sampler names."""
    excluded = {"dpm_fast", "dpm_adaptive", "lcm"}
    try:
        names = [a[len("sample_"):] for a in dir(kds) if a.startswith("sample_")]
    except Exception:
        names = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "uni_pc"]
    return sorted(n for n in names if n not in excluded)


class SamplerSPEED(io.ComfyNode):
    """Spectral Progressive Diffusion sampler node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSPEED",
            display_name="Sampler SPEED (Spectral Progressive Diffusion)",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Combo.Input(
                    "base_sampler", options=_list_samplers(), default="euler",
                    tooltip="Underlying ODE solver. Any comfy k_diffusion sampler is "
                            "supported. Multistep solvers restart "
                            "at each SPEED transition because we segment the schedule.",
                ),
                io.Combo.Input(
                    "transform", options=["dct", "dwt", "fft"], default="dct",
                    tooltip="Spectral basis used at each transition. DCT (default) "
                            "supports any scale ratio; DWT requires consecutive scales "
                            "to differ by exactly 2x; FFT accepts any ratio.",
                ),
                io.Combo.Input(
                    "mode", options=["delta_optimal", "manual"], default="delta_optimal",
                    tooltip="`delta_optimal` computes transitions from `scales`, "
                            "`delta`, and the selected power-spectrum preset. "
                            "`manual` uses user-specified sigma thresholds.",
                ),
                io.Combo.Input(
                    "model_preset", options=list(_PRESETS.keys()), default="flux",
                    tooltip="Power-spectrum preset for delta-optimal mode. `flux` and "
                            "`wan21` use measured (A, beta). `custom` uses "
                            "the manual A / beta inputs below.",
                ),
                io.String.Input(
                    "scales", default="0.5,1.0",
                    tooltip="Comma-separated resolution fractions ending at 1.0. Used in "
                            "delta_optimal mode. Example: `0.5,1.0` or `0.25,0.5,1.0`.",
                ),
                io.Float.Input(
                    "delta", default=0.01, min=1e-4, max=0.5, step=0.001,
                    tooltip="Noise-dominated tolerance. Smaller values transition later.",
                ),
                io.String.Input(
                    "manual_sigmas", default="0.85",
                    tooltip="Comma-separated sigma thresholds, one per transition (length "
                            "S-1 to match `scales`). Used in `manual` mode. Example "
                            "for scales=`0.25,0.5,1.0`: `0.95,0.85`.",
                ),
                io.Float.Input(
                    "spectrum_A", default=203.615097, min=0.0, max=1e6, step=0.001,
                    tooltip="Power-spectrum amplitude A (used when model_preset=custom).",
                ),
                io.Float.Input(
                    "spectrum_beta", default=1.915461, min=0.0, max=10.0, step=0.001,
                    tooltip="Power-spectrum decay exponent beta (used when model_preset=custom).",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=2**31 - 1, step=1,
                    tooltip="Seed for the spectral-noise padding at each transition.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, base_sampler, transform, mode, model_preset, scales, delta,
                manual_sigmas, spectrum_A, spectrum_beta, seed):
        preset = _PRESETS.get(model_preset)
        if preset is not None:
            A, beta = preset["A"], preset["beta"]
        else:
            A, beta = float(spectrum_A), float(spectrum_beta)

        parsed_scales = _parse_scales(scales)
        parsed_sigmas = _parse_sigmas(manual_sigmas) if mode == "manual" else []

        sampler = comfy.samplers.KSAMPLER(
            sample_speed,
            extra_options={
                "transform": transform,
                "base_sampler": base_sampler,
                "mode": mode,
                "scales": parsed_scales,
                "delta": float(delta),
                "spectrum_A": A,
                "spectrum_beta": beta,
                "manual_sigmas": parsed_sigmas,
                "seed": int(seed),
            },
        )
        return io.NodeOutput(sampler)
