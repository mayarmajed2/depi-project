"""Local Emotion Detection using Hugging Face Transformers.

Includes text emotion detection (using distilroberta) and facial emotion detection (using ViT).
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


# Lazy loaders for models to optimize start times and support Streamlit caching
_TEXT_PIPELINE = None
_FACE_PIPELINE = None


def get_device_and_dtype() -> tuple[int | str, Any]:
    """Helper to determine PyTorch device and dtype for local pipeline loading."""
    if torch is not None and torch.cuda.is_available():
        return 0, torch.float16
    return -1, torch.float32


class TextEmotionDetector:
    """Class to detect emotion from user text description."""

    def __init__(self, model_id: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_id = model_id

    def _get_pipeline(self):
        global _TEXT_PIPELINE
        if _TEXT_PIPELINE is None:
            from transformers import pipeline
            device, dtype = get_device_and_dtype()
            # Suppress unnecessary warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _TEXT_PIPELINE = pipeline(
                    "text-classification",
                    model=self.model_id,
                    device=device,
                    model_kwargs={"torch_dtype": dtype} if device >= 0 else {}
                )
        return _TEXT_PIPELINE

    def detect(self, text: str) -> dict[str, Any]:
        """Detect emotion from input text.
        
        Returns:
            Dict containing:
              - 'emotion': the dominant emotion (e.g. 'sadness', 'joy', 'anger', etc.)
              - 'scores': dictionary of all emotion probabilities
        """
        if not text.strip():
            return {"emotion": "neutral", "scores": {"neutral": 1.0}}

        try:
            pipe = self._get_pipeline()
            # Get all predictions (top_k=None returns list of all classes)
            results = pipe(text, top_k=None)
            
            # Map predictions to a dictionary
            scores = {item["label"]: float(item["score"]) for item in results}
            # j-hartmann model labels: joy, sadness, fear, anger, surprise, disgust, neutral
            dominant = max(scores, key=scores.get)
            
            # Standardize names to singular nouns or simple adjectives if needed
            label_mapping = {
                "joy": "happy",
                "sadness": "sad",
                "anger": "angry",
                "fear": "fearful",
                "surprise": "surprised",
                "disgust": "disgusted",
                "neutral": "neutral"
            }
            mapped_dominant = label_mapping.get(dominant, dominant)
            mapped_scores = {label_mapping.get(k, k): v for k, v in scores.items()}
            
            return {
                "emotion": mapped_dominant,
                "scores": mapped_scores,
                "raw_emotion": dominant
            }
        except Exception as e:
            # Robust error handling: fallback to neutral if something fails
            return {
                "emotion": "neutral",
                "scores": {"neutral": 1.0},
                "error": str(e)
            }


class FacialEmotionDetector:
    """Class to detect emotion from faces inside images using ViT."""

    def __init__(self, model_id: str = "dima806/facial_emotions_image_detection"):
        self.model_id = model_id

    def _get_pipeline(self):
        global _FACE_PIPELINE
        if _FACE_PIPELINE is None:
            from transformers import pipeline
            device, dtype = get_device_and_dtype()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _FACE_PIPELINE = pipeline(
                    "image-classification",
                    model=self.model_id,
                    device=device,
                    model_kwargs={"torch_dtype": dtype} if device >= 0 else {}
                )
        return _FACE_PIPELINE

    def detect(self, image: Image.Image) -> dict[str, Any]:
        """Detect emotion from face(s) in an image.
        
        Tries to locate faces with cv2 Haar cascade, crop them, and classify them.
        If no face is detected, runs classification on the full image as a fallback.
        """
        try:
            pipe = self._get_pipeline()
            
            # Step 1: Detect and crop faces if OpenCV is available
            face_images = []
            if cv2 is not None:
                # Convert PIL to CV2 grayscale
                img_rgb = np.array(image.convert("RGB"))
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(40, 40)
                )
                
                for (x, y, w, h) in faces:
                    # Crop face from PIL image with safety margin
                    margin = int(min(w, h) * 0.1)
                    left = max(0, x - margin)
                    top = max(0, y - margin)
                    right = min(image.width, x + w + margin)
                    bottom = min(image.height, y + h + margin)
                    
                    face_img = image.crop((left, top, right, bottom))
                    face_images.append(face_img)

            # Step 2: Classify cropped face(s) or fallback to full image
            target_image = image
            if face_images:
                # Use the first/largest detected face
                target_image = face_images[0]
                has_face = True
            else:
                has_face = False

            # Run transformers image classification pipeline
            results = pipe(target_image)
            
            # Model output format: list of dicts with 'label' and 'score'
            scores = {item["label"]: float(item["score"]) for item in results}
            dominant = max(scores, key=scores.get)

            # Standardize names
            label_mapping = {
                "angry": "angry",
                "disgust": "disgusted",
                "fear": "fearful",
                "happy": "happy",
                "sad": "sad",
                "surprise": "surprised",
                "neutral": "neutral"
            }
            mapped_dominant = label_mapping.get(dominant, dominant)
            mapped_scores = {label_mapping.get(k, k): v for k, v in scores.items()}

            return {
                "emotion": mapped_dominant,
                "scores": mapped_scores,
                "has_face": has_face,
                "num_faces_detected": len(face_images) if cv2 is not None else 0
            }

        except Exception as e:
            return {
                "emotion": "neutral",
                "scores": {"neutral": 1.0},
                "has_face": False,
                "num_faces_detected": 0,
                "error": str(e)
            }
