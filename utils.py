"""Shared spectral expansion and transition scheduling utilities."""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pywt
import torch
import yaml
from scipy.fft import dctn, idctn


def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum ``P(omega) = A * |omega|**(-beta)``.

    Args:
    - omega: Radial spatial frequency.
    - A: Power-law amplitude (fitted per VAE model in ``configs.yaml``).
    - beta: Power-law decay exponent (fitted per VAE model in ``configs.yaml``).

    Returns:
    - The power-spectrum value ``P(omega)``.
    """
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Return the activation time for one radial frequency ``omega``. This matches Eq. 9 in the paper.

    Args:
    - P_omega: Power-spectrum value ``P(omega)`` at the frequency of interest.
    - delta: Noise-dominated tolerance; smaller ``delta`` delays activation.

    Returns:
    - The activation time ``t_omega`` in ``(0, 1)``.
    """
    if delta >= 1.0:
        raise ValueError(
            f"delta={delta} >= 1, but we assume the error threshold is < 1."
        )
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def delta_optimal_transitions(
    scales: Sequence[float],
    delta: float,
    A: float,
    beta: float,
    H: int,
    W: int,
) -> List[float]:
    """Return transition times for adjacent scales. This matches Eq. 10 from the paper.

    Args:
    - scales: Strictly increasing scale list ending at 1.0. Example: ``[0.25, 0.5, 1.0]``.
    - delta: Noise-dominated tolerance passed to ``activation_time``.
    - A: Power-law amplitude.
    - beta: Power-law decay exponent.
    - H: Full-resolution latent height (sets ``omega_max = min(H, W) / 2``).
    - W: Full-resolution latent width (sets ``omega_max = min(H, W) / 2``).

    Returns:
    - List of transition times ``t*_i`` (length ``len(scales) - 1``).
    """
    validate_scales(scales)
    # Nyquist maximum representable frequency limit
    omega_max = min(H, W) / 2.0
    transitions: List[float] = []
    for i in range(len(scales) - 1):
        omega_i = scales[i] * omega_max
        transitions.append(activation_time(power_spectrum(omega_i, A, beta), delta))
    return transitions


def align_timestep(t: float, r: float) -> float:
    """Return the aligned flow-matching time after spectral noise expansion. This matches Eq. 6 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Resolution scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The aligned flow-matching time ``t_tilde``.
    """
    return t * kappa(t, r)


def kappa(t: float, r: float) -> float:
    """Return the state-rescaling factor after spectral noise expansion. This matches Eq. 5 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The state-rescaling factor ``kappa``.
    """
    return r / (1.0 + (r - 1.0) * t)


def _dct_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """DCT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the high-frequency coefficients, which is also the current flow matching time. 
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"DCT expand: cannot expand to target {target_hw} smaller than source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs_src = dctn(x_np[idx], type=2, norm="ortho")
        big = t * rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        big[:H_src, :W_src] = coeffs_src
        out[idx] = idctn(big, type=2, norm="ortho").astype(np.float32)
    return out


def _dwt_expand_np(x_np: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Haar wavelet spectral noise expansion. The target H, W is automatically two times the source H, W.

    Args:
    - x_np: Source array treated as the LL band; trailing two axes are spatial.
    - t: Noise amplitude for the LH/HL/HH detail bands, which is also the current flow matching time.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at twice the source resolution (float32).
    """
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    H_tgt, W_tgt = H_src * 2, W_src * 2
    rng = np.random.default_rng(seed)
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        LL = x_np[idx]
        LH = t * rng.standard_normal(LL.shape).astype(np.float32)
        HL = t * rng.standard_normal(LL.shape).astype(np.float32)
        HH = t * rng.standard_normal(LL.shape).astype(np.float32)
        out[idx] = pywt.waverec2(
            [LL, (LH, HL, HH)], "haar", mode="periodization"
        ).astype(np.float32)
    return out


def _fft_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """FFT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the outer (high-frequency) spectrum, which is also the current flow matching time.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"FFT expand: cannot expand to target {target_hw} smaller than source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    pad_h, pad_w = (H_tgt - H_src) // 2, (W_tgt - W_src) // 2
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        X_src = np.fft.fftshift(np.fft.fft2(x_np[idx], norm="ortho"))
        nr = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        ni = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        X_big = np.fft.fftshift(t * (nr + 1j * ni) / np.sqrt(2.0))
        X_big[pad_h:pad_h + H_src, pad_w:pad_w + W_src] = X_src
        out[idx] = np.fft.ifft2(np.fft.ifftshift(X_big), norm="ortho").real.astype(np.float32)
    return out


def spectral_expand_and_align(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
    r: float,
    t: float,
    transform: str,
    seed: int,
) -> Tuple[torch.Tensor, float]:
    """Expand ``x`` to ``target_hw`` and return the rescaled latent and aligned time.

    The caller is the single source of truth for spatial dimensions: it passes
    the already-resolved ``target_hw`` (e.g. snapped to even dims for FLUX) and
    the user-intended scale ratio ``r = s_{i + 1} / s_i`` used by the alignment
    maths (which must not drift with integer rounding of the dims).

    Args:
    - x: Current latent at the source resolution; trailing two axes are spatial.
    - target_hw: Target ``(height, width)`` to expand to (resolved by the caller).
    - r: Scale ratio ``s_{i + 1} / s_i`` of the transition.
    - t: Flow-matching time at the transition.
    - transform: Spectral basis, one of ``'dct'``, ``'dwt'``, ``'fft'``.
    - seed: Seed for the per-call random generator.

    Returns:
    - A ``(x_tilde, t_tilde)`` tuple: the rescaled and expanded latent and the
      aligned flow-matching time.
    """
    if transform not in ("dct", "dwt", "fft"):
        raise ValueError(f"transform must be 'dct'|'dwt'|'fft', got {transform!r}")

    H_tgt, W_tgt = target_hw
    H_src, W_src = x.shape[-2], x.shape[-1]
    x_np = x.detach().cpu().float().numpy()
    if transform == "dwt":
        if abs(r - 2.0) > 1e-6:
            raise ValueError(
                f"DWT requires a 2x scale ratio between consecutive scales; "
                f"got r = s_next/s_i = {r:.4f}. "
                f"Use --transform dct or --transform fft for non-dyadic scales."
            )
        # Haar IDWT is structurally 2x
        if H_tgt != 2 * H_src or W_tgt != 2 * W_src:
            raise ValueError(
                f"DWT requires H_tgt = 2·H_src and W_tgt = 2·W_src; got "
                f"({H_src}, {W_src}) -> ({H_tgt}, {W_tgt})."
            )
        expanded = _dwt_expand_np(x_np, t, seed)
    elif transform == "dct":
        expanded = _dct_expand_np(x_np, (H_tgt, W_tgt), t, seed)
    else:  # fft
        expanded = _fft_expand_np(x_np, (H_tgt, W_tgt), t, seed)

    rescaled = (kappa(t, r) * expanded).astype(np.float32)
    x_tilde = torch.from_numpy(rescaled).to(device=x.device, dtype=x.dtype)
    return x_tilde, align_timestep(t, r)


def find_first_step_below(sigmas: Iterable[float], threshold: float) -> int:
    """Return the first step index whose sigma (step timestamp) is below ``threshold (transition timestep)``.

    Args:
    - sigmas: Scheduler sigma sequence (length ``n_steps + 1``).
    - threshold: Sigma threshold; the first step at or below it is returned.

    Returns:
    - The first step index with ``sigma <= threshold``, or ``n_steps`` if none.
    """
    sigmas = list(sigmas)
    n_steps = len(sigmas) - 1
    for i in range(n_steps):
        s = sigmas[i].item() if hasattr(sigmas[i], "item") else float(sigmas[i])
        if s <= threshold:
            return i
    return n_steps


def reset_scheduler_state(scheduler, step_index: int) -> None:
    """Reset solver buffers after a transition.

    Args:
    - scheduler: Diffusers scheduler whose solver state is reset in place.
    - step_index: Step index to set ``scheduler._step_index`` to.

    Returns:
    - None; the scheduler is mutated in place.
    """
    if hasattr(scheduler, "model_outputs"):
        order = getattr(scheduler.config, "solver_order", 1)
        scheduler.model_outputs = [None] * order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    scheduler._step_index = step_index


def validate_scales(scales: Sequence[float]) -> None:
    """Validate a strictly increasing resolution scale list ending at 1.0.

    Args:
    - scales: Scale list to validate; each value in ``(0, 1]``, strictly
      increasing, ending at ``1.0``.

    Returns:
    - None; raises ``ValueError`` if the scales are invalid.
    """
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def parse_scales(tokens, full_height: int) -> List[float]:
    """Parse a scale spec into ``[s_1, ..., s_S = 1.0]``.

    Tokens may be space- or comma-separated and take any of three forms,
    freely mixed:
      - decimal scales -- ``0.5`` ``0.37``
      - fraction scales -- ``1/2`` ``2/3``
      - stage heights in pixels -- ``480 720`` (triggered when any value
        exceeds 1); the last must equal ``full_height`` and every value is
        divided by it to recover the scale.

    Args:
    - tokens: Scale tokens (strings or numbers); each may itself be a
      comma-separated group.
    - full_height: Configured full-resolution height, used to convert the
      pixel-height form to scales and to validate its last value.

    Returns:
    - The validated scale list ``[s_1, ..., s_S = 1.0]``.
    """
    flat: List[str] = []
    for t in tokens:
        flat.extend(str(t).split(","))

    def _one(tok: str) -> float:
        tok = tok.strip()
        if "/" in tok:
            num, den = tok.split("/", 1)
            return float(num) / float(den)
        return float(tok)

    values = [_one(t) for t in flat if t.strip()]
    if not values:
        raise ValueError("--scales is empty; supply at least one value.")
    if max(values) > 1.0 + 1e-6:
        # Resolution mode: tokens are the pixel height of each stage.
        if abs(values[-1] - full_height) > 1e-6:
            raise ValueError(
                f"the last stage resolution {values[-1]:g} must equal the "
                f"configured height {full_height}; otherwise pass scales in (0, 1]."
            )
        values = [v / full_height for v in values]
    validate_scales(values)
    return values


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    """Recursively expand ``${ENV_VAR}`` placeholders in a config value.

    Args:
    - value: A str, dict, or list parsed from YAML; other types pass through.

    Returns:
    - The value with every ``${VAR}`` placeholder expanded (same structure
      as the input).
    """
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            """Substitute one ``${VAR}`` match with its environment value.

            Args:
            - m: Regex match whose group 1 is the environment variable name.

            Returns:
            - The environment-variable value for the matched name.
            """
            var = m.group(1)
            if var not in os.environ:
                raise KeyError(
                    f"Environment variable {var!r} referenced in configs.yaml "
                    "is not set."
                )
            return os.environ[var]
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(yaml_path: str | Path, model_key: str) -> dict:
    """Load a model config and expand ``${ENV_VAR}`` placeholders.

    Args:
    - yaml_path: Path to the ``configs.yaml`` file.
    - model_key: Top-level model key to select (e.g. ``'flux'``, ``'wan21'``).

    Returns:
    - The selected model config dict with ``${ENV_VAR}`` placeholders expanded.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if model_key not in data:
        raise KeyError(f"model {model_key!r} not in {yaml_path}; have {list(data)}")
    return _expand_env(data[model_key])


def load_prompts(args) -> List[str]:
    """Collect prompts from ``--prompts``, ``--prompt_txt``, or ``--prompt_csv``.

    Args:
    - args: Parsed argparse namespace.

    Returns:
    - The list of prompt strings.
    """
    if args.prompts:
        return list(args.prompts)
    if args.prompt_txt:
        return [ln.strip() for ln in open(args.prompt_txt) if ln.strip()]
    return [row["prompt"] for row in csv.DictReader(open(args.prompt_csv))]
