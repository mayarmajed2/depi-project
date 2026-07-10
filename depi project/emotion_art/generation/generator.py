"""Local Stable Diffusion image generation optimized for RTX 3050.

Implements float16, attention slicing, and model CPU offloading to fit within
smaller VRAM budgets (4GB - 8GB) and support fast offline generations.
"""

from __future__ import annotations

import os
from typing import Optional, Any
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline, LCMScheduler

try:
    from emotion_art.prompts.constants import NEGATIVE_PROMPT
except Exception:
    NEGATIVE_PROMPT = (
        "low quality, blurry, bad anatomy, extra fingers, distorted face, watermark, "
        "text, duplicate, cropped, ugly, jpeg artifacts"
    )

# Default models: standard runwayml SD 1.5, or a lightweight/turbo model.
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Lazy loader global
_PIPELINE: Optional[StableDiffusionPipeline] = None
_CURRENT_MODEL_ID: Optional[str] = None


def get_sd_model_id() -> str:
    """Resolve model ID from environment or fallback to default."""
    return (os.environ.get("SD_MODEL_ID") or DEFAULT_MODEL_ID).strip()


def get_hf_token() -> Optional[str]:
    """Retrieve HF Token for gated models."""
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t or None


def get_generator_pipeline() -> StableDiffusionPipeline:
    """Load and optimize Stable Diffusion pipeline for the current hardware (especially RTX 3050)."""
    global _PIPELINE, _CURRENT_MODEL_ID
    model_id = get_sd_model_id()

    # Re-use already loaded pipeline if model ID matches
    if _PIPELINE is not None and _CURRENT_MODEL_ID == model_id:
        return _PIPELINE

    # Clear old pipeline to free VRAM
    if _PIPELINE is not None:
        _PIPELINE = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Determine precision and device
    is_cuda = torch is not None and torch.cuda.is_available()
    dtype = torch.float16 if is_cuda else torch.float32
    token = get_hf_token()

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        kwargs["token"] = token

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except OSError as e:
        raise RuntimeError(
            f"Could not load Stable Diffusion model '{model_id}' from Hugging Face Hub.\n"
            f"Please verify your model ID and that you have accepted its terms on HF if it is gated.\n"
            f"Error details: {e}"
        ) from e

    # Configure scheduler (LCM / DPM Multi-step for fast inference)
    # Check if we are using an LCM / Turbo model or standard SD
    is_turbo_or_lcm = "turbo" in model_id.lower() or "lcm" in model_id.lower()
    
    if is_turbo_or_lcm:
        # Use LCMScheduler for LCM/Turbo models (needs only 1-4 steps!)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        # Use high quality EulerAncestralDiscreteScheduler for normal SD to avoid index errors
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Optimization settings for RTX 3050 & small VRAM budgets
    if is_cuda:
        # 1. Shift pipeline to GPU
        pipe = pipe.to("cuda")
        
        # 2. Enable attention slicing to split computations and save substantial VRAM
        pipe.enable_attention_slicing()
        
        # 3. Optional: Enable model CPU offloading if VRAM is extremely limited (e.g. 4GB laptop RTX 3050)
        # Note: sequential/model CPU offload slows down inference slightly but prevents OOMs.
        # We enable model CPU offloading by default if VRAM budget is small (< 6GB)
        # To avoid complex check, we'll let the user toggle CPU offload in Streamlit or enable attention slicing first.
        try:
            # Try to use xformers if available
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass # fallback to attention slicing
    else:
        pipe = pipe.to("cpu")

    _PIPELINE = pipe
    _CURRENT_MODEL_ID = model_id
    return _PIPELINE


def generate_artwork_local(
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    height: int = 512,
    width: int = 512,
    enable_cpu_offload: bool = False
) -> Image.Image:
    """Generate artwork locally using the optimized Stable Diffusion pipeline.
    
    Args:
        prompt: Engineered visual prompt.
        negative_prompt: Negative prompt elements.
        num_inference_steps: Scheduler steps (lower is faster; LCM/Turbo use ~4, standard SD uses ~25-30).
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed for repeatability.
        height: Output image height.
        width: Output image width.
        enable_cpu_offload: If True, enables model CPU offloading to save VRAM on tight budgets.
        
    Returns:
        Generated PIL Image.
    """
    pipe = get_generator_pipeline()
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    
    # Setup seed
    if seed is None:
        import sys
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    # Configure CPU offload toggle on-the-fly
    if device == "cuda":
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            # Put pipeline fully back on GPU
            pipe.to("cuda")

    # Use standard negative prompt if not overridden
    neg_p = negative_prompt if negative_prompt is not None else NEGATIVE_PROMPT
    
    # Auto-adjust steps if SD Turbo or LCM is loaded to save time
    model_id = get_sd_model_id().lower()
    if ("turbo" in model_id or "lcm" in model_id) and num_inference_steps > 10:
        # Force lower steps for Turbo models to make it lightning fast
        num_inference_steps = 4
        guidance_scale = 0.0 # Turbo models generally work best with 0.0 or 1.0 guidance

    # Execute inference
    result = pipe(
        prompt=prompt,
        negative_prompt=neg_p,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width
    )
    
    return result.images[0]
