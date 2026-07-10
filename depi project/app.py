"""Emotion-Aware Artwork Generation — Local & Offline Streamlit UI

Uses local HuggingFace Transformers and Stable Diffusion models.
Optimized for RTX 3050 and other consumer GPUs.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from PIL import Image

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# Local modules
from emotion_art.analysis.emotion_detector import TextEmotionDetector, FacialEmotionDetector
from emotion_art.analysis.image_captioner import ImageCaptioner
from emotion_art.generation.generator import generate_artwork_local, get_sd_model_id
from emotion_art.prompts.prompt_engineer import PromptEngineer
from emotion_art.explanation.explainer import ArtExplainer
from emotion_art.utils.color_analyzer import extract_dominant_colors
from emotion_art.utils.io_utils import save_image_png
from emotion_art.utils.visualizations import (
    analysis_text_for_wordcloud,
    color_emphasis_overlay,
    highlight_faces,
    word_cloud_image,
)

# Import NEGATIVE_PROMPT with fallback protection
try:
    from emotion_art.prompts.constants import NEGATIVE_PROMPT
except Exception:
    NEGATIVE_PROMPT = (
        "low quality, blurry, bad anatomy, extra fingers, distorted face, watermark, "
        "text, duplicate, cropped, ugly, jpeg artifacts"
    )

# --- Caching model loaders for Streamlit ---
@st.cache_resource
def load_text_detector() -> TextEmotionDetector:
    return TextEmotionDetector()

@st.cache_resource
def load_facial_detector() -> FacialEmotionDetector:
    return FacialEmotionDetector()

@st.cache_resource
def load_image_captioner() -> ImageCaptioner:
    return ImageCaptioner()


def inject_custom_css() -> None:
    """Inject custom styles for a premium dark/glassmorphic look."""
    st.markdown(
        """
        <style>
        /* Modern font and colors */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        .main .block-container {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Premium Header Styling */
        .title-container {
            background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }
        
        /* Glassmorphic card styling for explanations */
        .critique-card {
            background: rgba(255, 255, 255, 0.05);
            border-left: 5px solid #6C63FF;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-left: 5px solid #6C63FF;
        }
        
        /* Color Swatch Container */
        .color-palette-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 1.5rem 0;
        }
        
        .color-swatch {
            width: 80px;
            height: 80px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            border: 2px solid rgba(255,255,255,0.2);
            transition: transform 0.2s;
        }
        
        .color-swatch:hover {
            transform: scale(1.08);
        }
        
        .color-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-align: center;
            margin-top: 5px;
        }
        
        .color-percent {
            font-size: 0.7rem;
            color: #888;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Emotion-to-Art Local Multimodal AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    
    # Header
    st.markdown('<div class="title-container">Emotion-to-Art AI</div>', unsafe_allow_html=True)
    st.markdown(
        "**Local & Offline Multimodal AI System** — Detects emotions from text or images, "
        "captions scenes, and generates high-fidelity local artwork explaining the artistic decisions."
    )
    
    # Check GPU availability
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"⚡ GPU Detected: **{gpu_name}** (Running in CUDA mode)")
    else:
        st.sidebar.warning("⚠️ No GPU detected. Running on CPU mode (generation will be slower).")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("🎨 Generation Parameters")
        
        # Style selector
        style_choice = st.selectbox(
            "Artistic Style Template",
            options=["fused", "van_gogh", "monet", "da_vinci", "expressionism"],
            format_func=lambda x: {
                "fused": "Fused Masters (Van Gogh + Monet + Da Vinci)",
                "van_gogh": "Vincent van Gogh (Impasto / Swirls)",
                "monet": "Claude Monet (Atmospheric Impressionism)",
                "da_vinci": "Leonardo da Vinci (Sfumato / Chiaroscuro)",
                "expressionism": "Expressionism (Raw / Exaggerated Emotion)"
            }.get(x, x),
            index=0
        )
        
        steps = st.slider("Inference steps", 5, 50, 25, help="More steps = higher detail, but slower. Recommend 20-30 for SD v1.5, 4 for SD Turbo.")
        guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, help="How strongly the model matches your prompt.")
        seed_in = st.text_input("Manual Seed (optional, integer)", "")
        
        st.markdown("---")
        st.header("⚙️ RTX 3050 & VRAM Optimizations")
        
        # GPU low memory optimizations
        enable_cpu_offload = st.checkbox(
            "Enable Model CPU Offload",
            value=False,
            help="Saves substantial VRAM by unloading sub-modules from GPU when inactive. Recommended if encountering Out-Of-Memory errors."
        )
        
        # Viz choices
        st.markdown("---")
        st.header("🔍 Visual Analysis Overlays")
        show_heat = st.checkbox("Saturation Emphasis Overlay", value=True)
        show_faces = st.checkbox("Face Highlights", value=True)
        show_wc = st.checkbox("Explanation Word Cloud", value=True)
        
        # Show configured local model ID
        st.markdown("---")
        st.caption(f"Loaded Stable Diffusion model ID: `{get_sd_model_id()}`")

    # --- Input Section ---
    st.markdown("### 📥 Step 1: Input Expression")
    
    input_tab1, input_tab2 = st.tabs(["📝 Text Description", "📷 Image Upload"])
    
    input_mode = "text"
    user_text = ""
    uploaded_file = None
    
    with input_tab1:
        user_text = st.text_area(
            "How do you feel or what scene are you imagining?",
            value="I feel quiet and a bit lonely watching the rain fall outside.",
            placeholder="Describe your emotion, day, or scene in detail...",
            key="text_input_area"
        )
        
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload an image or portrait/selfie (we will detect emotions from faces and analyze the scene context)",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            input_mode = "image"
            st.image(uploaded_file, caption="Uploaded Image", width=300)

    # --- Trigger Generation ---
    if st.button("Generate Emotional Artwork", type="primary", use_container_width=True):
        # 1. Input validation
        if input_mode == "text" and not user_text.strip():
            st.error("Please enter a text description of your feelings or scene.")
            return
        if input_mode == "image" and uploaded_file is None:
            st.error("Please upload an image to analyze.")
            return
            
        seed_val = None
        if seed_in.strip():
            try:
                seed_val = int(seed_in.strip())
            except ValueError:
                st.error("Seed must be an integer.")
                return

        # 2. Run Local Multi-Modal Pipeline
        status_box = st.status("Initializing Local AI Pipeline...", expanded=True)
        
        # Load local models
        status_box.write("Loading Transformers Emotion Detectors...")
        text_detector = load_text_detector()
        facial_detector = load_facial_detector()
        
        status_box.write("Loading Image Captioning Model...")
        image_captioner = load_image_captioner()
        
        detected_emotion = "neutral"
        emotion_scores = {"neutral": 1.0}
        visual_caption = ""
        has_face = False
        
        # Pipeline: Step 1 & 2 - Emotion Detection & Context analysis
        if input_mode == "text":
            status_box.write("Running Text Emotion Classification...")
            analysis_res = text_detector.detect(user_text)
            detected_emotion = analysis_res["emotion"]
            emotion_scores = analysis_res["scores"]
            visual_caption = f"Conceptual visualization of: {user_text}"
        else:
            status_box.write("Running Image Scene Understanding & Facial Analysis...")
            pil_image = Image.open(uploaded_file).convert("RGB")
            
            # Detect emotion from face
            face_res = facial_detector.detect(pil_image)
            detected_emotion = face_res["emotion"]
            emotion_scores = face_res["scores"]
            has_face = face_res.get("has_face", False)
            
            # Generate BLIP Caption for context
            visual_caption = image_captioner.caption(pil_image)
            
        status_box.write(f"Detected dominant emotion: **{detected_emotion.upper()}**")
        if visual_caption:
            status_box.write(f"Generated scene context: *\"{visual_caption}\"*")
            
        # Pipeline: Step 3 - Prompt Engineering
        status_box.write("Engineering artistic prompt...")
        prompt_eng = PromptEngineer(default_style=style_choice)
        engineered_prompt = prompt_eng.engineer_prompt(
            emotion=detected_emotion,
            context_caption=visual_caption if input_mode == "image" else user_text,
            style_choice=style_choice
        )
        
        # Pipeline: Step 4 - Stable Diffusion Generation
        status_box.write("Loading Stable Diffusion Pipeline to GPU...")
        try:
            # Generate artwork
            status_box.write("Stable Diffusion: Generating artwork...")
            generated_img = generate_artwork_local(
                prompt=engineered_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                seed=seed_val,
                enable_cpu_offload=enable_cpu_offload
            )
        except Exception as e:
            status_box.update(label="Artwork Generation Failed", state="error")
            st.exception(e)
            return

        # Pipeline: Step 5 - Color Feature Extraction (CV2 K-Means)
        status_box.write("Extracting dominant color palette using K-Means...")
        color_palette = extract_dominant_colors(generated_img, k=5)
        
        # Pipeline: Step 6 - AI Explanation
        status_box.write("Generating artistic explanation...")
        explainer = ArtExplainer()
        explanation_md = explainer.generate_explanation(
            emotion=detected_emotion,
            dominant_colors=color_palette,
            style_choice=style_choice,
            caption_context=visual_caption if input_mode == "image" else None
        )
        
        # Save image
        saved_path = save_image_png(generated_img, prefix="emotion_art")
        
        # Close status loader
        status_box.update(label="Emotional Artwork Complete!", state="complete", expanded=False)
        
        # --- UI DISPLAY OF RESULTS ---
        st.markdown("### 📤 Step 2: Generated Masterpiece")
        
        out_col1, out_col2 = st.columns([1, 1])
        
        with out_col1:
            st.image(generated_img, caption="Generated local artwork", use_container_width=True)
            
            # Display dominant color swatches
            st.write("**Extracted Dominant Color Palette:**")
            swatch_html = '<div class="color-palette-container">'
            for col in color_palette:
                hex_val = col["hex"]
                pct = col["percentage"] * 100
                swatch_html += f"""
                <div>
                    <div class="color-swatch" style="background-color: {hex_val};"></div>
                    <div class="color-label">{hex_val}</div>
                    <div class="color-percent">{pct:.1f}%</div>
                </div>
                """
            swatch_html += "</div>"
            st.markdown(swatch_html, unsafe_allow_html=True)
            
        with out_col2:
            st.metric(label="Detected Emotion", value=detected_emotion.upper())
            
            # Draw probability bar chart for emotion scores
            st.write("**Emotion Confidence Scores:**")
            # Sort scores to make chart look nice
            sorted_scores = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(sorted_scores)
            
            # Show engineered prompt in a code box
            with st.expander("Engineered Stable Diffusion Prompt"):
                st.code(engineered_prompt, language="text")
                st.caption(f"Negative prompt used: `{NEGATIVE_PROMPT}`")
                
        # Full width critique
        st.markdown(f'<div class="critique-card">{explanation_md}</div>', unsafe_allow_html=True)
        
        # Build payload for JSON download
        payload = {
            "detected_emotion": detected_emotion,
            "visual_context": visual_caption,
            "engineered_prompt": engineered_prompt,
            "color_palette": [{"hex": c["hex"], "rgb": c["rgb"], "percentage": c["percentage"]} for c in color_palette],
            "saved_image_path": saved_path
        }
        
        dl_json = json.dumps(payload, indent=2, ensure_ascii=False)
        st.download_button("💾 Download Art Analysis JSON", data=dl_json, file_name="emotion_art_local_result.json")
        
        # --- Visualization Overlays ---
        st.markdown("### 🔍 Step 3: Computer Vision & Analytics")
        tabs = []
        if show_heat:
            tabs.append("Color Saturation Map")
        if show_faces:
            tabs.append("Face Outlines")
        if show_wc:
            tabs.append("Art Critique Word Cloud")
            
        if tabs:
            ts = st.tabs(tabs)
            ti = 0
            
            if show_heat:
                with ts[ti]:
                    st.caption("Highlights warm hues and saturation weights across the canvas.")
                    st.image(color_emphasis_overlay(generated_img), use_container_width=True)
                ti += 1
                
            if show_faces:
                with ts[ti]:
                    st.caption("Locates human faces using Haar Cascade classifiers on the uploaded image (if available) or the generated image.")
                    target_for_faces = Image.open(uploaded_file).convert("RGB") if input_mode == "image" else generated_img
                    face_img, n = highlight_faces(target_for_faces)
                    if n == 0:
                        st.info("No faces detected in the image.")
                    st.image(face_img, use_container_width=True)
                ti += 1
                
            if show_wc:
                with ts[ti]:
                    st.caption("Visual distribution of words from the AI Art Critique.")
                    # Use explanation markdown for word cloud
                    wc = word_cloud_image(analysis_text_for_wordcloud({"explanation": explanation_md}))
                    if wc is None:
                        st.info("Word Cloud is unavailable (install wordcloud package).")
                    else:
                        st.image(wc, use_container_width=True)
                ti += 1

    with st.expander("ℹ️ About the Offline System"):
        st.markdown(
            """
            This application has been refactored into a fully offline local multimodal AI system.
            
            **Local AI Pipelines:**
            1. **Text Emotion Classifier:** `j-hartmann/emotion-english-distilroberta-base` (maps 7 emotions)
            2. **Facial Emotion Classifier:** `dima806/facial_emotions_image_detection` (ViT classifier)
            3. **Image Captioning (BLIP):** `Salesforce/blip-image-captioning-base` (context builder)
            4. **Stable Diffusion:** `runwayml/stable-diffusion-v1-5` or custom model ID set in `.env`
            5. **Color Analyzer:** OpenCV K-Means cluster palette extractor
            6. **AI Explanation:** Custom expert system mapping colors and styles to art theory
            
            *No external API calls are made. All operations occur locally in PyTorch.*
            """
        )


if __name__ == "__main__":
    main()
