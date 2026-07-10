"""Utils package exposing file I/O, visualization overlays, and color analysis tools."""

from __future__ import annotations

from emotion_art.utils.io_utils import save_image_png, default_output_dir
from emotion_art.utils.visualizations import (
    color_emphasis_overlay,
    highlight_faces,
    word_cloud_image,
    analysis_text_for_wordcloud,
)
from emotion_art.utils.color_analyzer import extract_dominant_colors

__all__ = [
    "save_image_png",
    "default_output_dir",
    "color_emphasis_overlay",
    "highlight_faces",
    "word_cloud_image",
    "analysis_text_for_wordcloud",
    "extract_dominant_colors",
]
