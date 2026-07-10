"""Optional overlays: color emphasis map, face highlights, word cloud.

Moved to utils/ to follow the new project structure.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None  # type: ignore


def color_emphasis_overlay(
    image: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Semi-transparent heatmap: combines saturation and warm hue emphasis (not ML saliency).
    """
    rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    h, s, _v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Warm hues in matplotlib HSV: red/orange/yellow ~0–0.15
    warm = np.clip(1.0 - np.minimum(np.abs(h - 0.05), np.abs(h - 1.0)) / 0.2, 0, 1)
    intensity = np.clip(s * 0.6 + warm * 0.4, 0, 1)
    import matplotlib
    cmap = matplotlib.colormaps["magma"]
    heat = cmap(intensity)[:, :, :3]
    heat_u8 = (heat * 255).astype(np.uint8)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    blend = (alpha * heat_u8.astype(np.float32) + (1 - alpha) * rgb_u8.astype(np.float32)).astype(
        np.uint8
    )
    return Image.fromarray(blend)


def highlight_faces(image: Image.Image) -> Tuple[Image.Image, int]:
    """
    Draw rectangles on detected faces (Haar cascade). Returns (annotated_image, num_faces).
    If OpenCV missing, returns a copy of the original and 0.
    """
    if cv2 is None:
        return image.copy(), 0
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    out = rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 60, 60), 3)
    return Image.fromarray(out), len(faces)


def analysis_text_for_wordcloud(analysis: dict) -> str:
    parts: List[str] = []
    for k, v in analysis.items():
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def word_cloud_image(
    text: str,
    width: int = 800,
    height: int = 400,
) -> Optional[Image.Image]:
    if WordCloud is None or not text.strip():
        return None
    clean = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=80,
        colormap="viridis",
    ).generate(clean)
    arr = wc.to_array()
    return Image.fromarray(arr)
