"""ComfyUI custom-node entry-point for Spectral Progressive Diffusion."""

from .speed_sampler import SamplerSPEED

NODE_CLASS_MAPPINGS = {
    "SamplerSPEED": SamplerSPEED,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSPEED": "Sampler SPEED (Spectral Progressive Diffusion)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
