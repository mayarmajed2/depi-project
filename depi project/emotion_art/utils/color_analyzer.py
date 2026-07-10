"""Dominant color extraction utility using OpenCV K-Means.

Analyzes generated images to extract dominant color palettes (RGB, Hex, and weights)
for style validation and the AI explanation system.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple

try:
    import cv2
except ImportError:
    cv2 = None


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB integers to a hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_dominant_colors(
    image: Image.Image,
    k: int = 5,
    max_resize: int = 150
) -> List[Dict[str, Any]]:
    """Extract dominant colors from a PIL Image.
    
    Args:
        image: PIL Image to analyze.
        k: Number of dominant colors to extract.
        max_resize: Size to resize the image to for faster processing.
        
    Returns:
        List of dictionaries containing:
          - 'rgb': Tuple of (R, G, B) integers
          - 'hex': Hex color string (e.g. '#a123bc')
          - 'percentage': Fraction of image pixels corresponding to this color (0.0 to 1.0)
    """
    # Fallback if CV2 is not available or K-Means fails
    if cv2 is None:
        return _fallback_extract(image, k)

    try:
        # Resize image for speed
        img_resized = image.copy()
        img_resized.thumbnail((max_resize, max_resize))
        
        # Convert PIL to CV2 BGR
        img_np = np.array(img_resized.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Reshape to 2D list of pixels
        pixels = img_bgr.reshape(-1, 3).astype(np.float32)
        
        # Define criteria and run K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
        
        # Count frequency of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        
        # Combine centers, counts, and convert to RGB
        colors_info = []
        for label, count in zip(unique_labels, counts):
            bgr_color = centers[label]
            # Convert BGR to RGB
            rgb_color = (int(round(bgr_color[2])), int(round(bgr_color[1])), int(round(bgr_color[0])))
            percentage = float(count) / total_pixels
            hex_str = rgb_to_hex(*rgb_color)
            
            colors_info.append({
                "rgb": rgb_color,
                "hex": hex_str,
                "percentage": percentage
            })
            
        # Sort colors by percentage descending
        colors_info.sort(key=lambda x: x["percentage"], reverse=True)
        return colors_info

    except Exception:
        return _fallback_extract(image, k)


def _fallback_extract(image: Image.Image, k: int = 5) -> List[Dict[str, Any]]:
    """Simple color extraction fallback using PIL's built-in color quantization."""
    try:
        # Resize image and convert to Palette mode (P) which quantizes colors
        img_resized = image.copy()
        img_resized.thumbnail((100, 100))
        img_quantized = img_resized.quantize(colors=k)
        
        # Get dominant colors list
        palette = img_quantized.getpalette()
        # count colors
        color_counts = img_quantized.getcolors()
        if not color_counts:
            return []
            
        total_pixels = sum(count for count, _ in color_counts)
        colors_info = []
        
        for count, idx in color_counts:
            # Each palette color is 3 values (R, G, B)
            r = palette[idx * 3]
            g = palette[idx * 3 + 1]
            b = palette[idx * 3 + 2]
            rgb_color = (int(r), int(g), int(b))
            percentage = float(count) / total_pixels
            hex_str = rgb_to_hex(*rgb_color)
            
            colors_info.append({
                "rgb": rgb_color,
                "hex": hex_str,
                "percentage": percentage
            })
            
        colors_info.sort(key=lambda x: x["percentage"], reverse=True)
        return colors_info
        
    except Exception:
        # Absolute basic fallback
        return [
            {"rgb": (128, 128, 128), "hex": "#808080", "percentage": 1.0}
        ]
