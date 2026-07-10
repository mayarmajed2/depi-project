"""Local Image Captioning/Understanding using BLIP.

Uses Salesforce/blip-image-captioning-base to generate descriptive text
from user uploaded images.
"""

from __future__ import annotations

import os
from PIL import Image
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None


# Lazy loader variables
_BLIP_PROCESSOR = None
_BLIP_MODEL = None


def get_device_and_dtype() -> tuple[str, Any]:
    """Helper to determine PyTorch device and dtype for BLIP loading."""
    if torch is not None and torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class ImageCaptioner:
    """Class to understand and describe an uploaded image using BLIP."""

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        self.model_id = model_id

    def _load_model(self):
        global _BLIP_PROCESSOR, _BLIP_MODEL
        if _BLIP_PROCESSOR is None or _BLIP_MODEL is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            device, dtype = get_device_and_dtype()
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processor = BlipProcessor.from_pretrained(self.model_id)
                
                # Load with float16 if cuda to save VRAM
                model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    torch_dtype=dtype
                ).to(device)
                
                _BLIP_PROCESSOR = processor
                _BLIP_MODEL = model
                
        return _BLIP_PROCESSOR, _BLIP_MODEL

    def caption(self, image: Image.Image, prompt: str = "a painting of") -> str:
        """Generate descriptive caption of the image.
        
        Args:
            image: PIL Image to caption.
            prompt: Text prompt prefix to guide BLIP's generation (e.g. 'a photo of').
        
        Returns:
            Description string.
        """
        try:
            processor, model = self._load_model()
            device, _ = get_device_and_dtype()
            
            # Prepare image
            inputs = processor(image, text=prompt, return_tensors="pt").to(device, dtype=model.dtype)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
            
            # Decode output
            caption_text = processor.decode(out[0], skip_special_tokens=True)
            return caption_text.strip()
            
        except Exception as e:
            return f"an uploaded image (captioning failed: {str(e)})"
