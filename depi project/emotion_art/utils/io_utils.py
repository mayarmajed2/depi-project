"""Save generated images and resolve paths for JSON output.

Moved to utils/ to follow the new project structure.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import Image


def default_output_dir() -> Path:
    # Resolve relative to the project root (parent of parent of parent of this file)
    # this file: emotion_art/utils/io_utils.py
    # parent: emotion_art/utils/
    # parent.parent: emotion_art/
    # parent.parent.parent: project_root/
    base = Path(__file__).resolve().parent.parent.parent
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_image_png(image: Image.Image, prefix: str = "artwork") -> str:
    """
    Save PNG under outputs/ with a timestamp. Returns absolute path as string.
    """
    out_dir = default_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_{ts}.png"
    image.save(path, format="PNG")
    return str(path.resolve())
