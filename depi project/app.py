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
    """Inject custom styles for a premium dark dashboard look."""
    st.markdown(
        """
        <style>
        /* Force Dark Mode Colors and Hide Streamlit UI elements */
        :root {
            --primary-color: #8B5CF6;
            --background-color: #0b0e14;
            --secondary-background-color: #161b22;
            --text-color: #e5e7eb;
        }
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Hide standard Sidebar and Top Bar completely */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        header {visibility: hidden !important;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Remove top padding */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
            max-width: 95% !important;
        }
        
        /* Modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Top Navigation Bar */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 30px;
        }
        .top-nav h2 {
            margin: 0;
            font-weight: 800;
            font-size: 2rem;
            color: #fff;
            letter-spacing: 2px;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        .nav-btn {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .nav-btn-primary {
            background: var(--primary-color);
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        /* Layout Column Borders */
        [data-testid="column"]:first-child {
            background-color: var(--secondary-background-color);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.05);
            height: 100%;
        }
        
        /* Primary Buttons (Generate) */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #8B5CF6 0%, #6C63FF 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-weight: bold;
            font-size: 1.2rem;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
            margin-top: 15px;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        }
        
        /* Text Areas and Inputs */
        .stTextArea textarea {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(139, 92, 246, 0.5);
            border-radius: 16px;
            color: white;
            font-size: 1.5rem;
            padding: 1.5rem;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
        }
        .stTextArea textarea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
        }
        .stTextArea label {
            font-size: 1rem !important;
            color: #8b949e !important;
        }
        
        /* Dashboard Card Wrappers */
        .dashboard-header {
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #fff;
            margin-bottom: 20px;
            font-weight: 700;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 3rem !important;
            color: var(--primary-color) !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #8b949e !important;
            text-transform: uppercase;
        }
        
        /* Glassmorphic card styling for explanations */
        .critique-card {
            background: rgba(22, 27, 34, 0.8);
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-left: 5px solid #8B5CF6;
        }
        
        /* Color Swatch Container */
        .color-palette-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 1.5rem 0;
        }
        .color-swatch {
            width: 60px;
            height: 60px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }
        .color-swatch:hover {
            transform: scale(1.08);
        }
        .color-label { font-size: 0.7rem; font-weight: 600; text-align: center; margin-top: 5px; color: #c9d1d9;}
        .color-percent { font-size: 0.6rem; color: #8b949e; text-align: center; }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="AURORA AI Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    inject_custom_css()
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "new"
    
    # Custom Dashboard Header using Columns for interactivity
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    head1, head2, head3 = st.columns([7, 1, 1], gap="small")
    with head1:
        st.markdown('<h2 style="margin-top:-5px;"><span style="color:#8B5CF6">▲</span> AURORA AI</h2>', unsafe_allow_html=True)
    with head2:
        if st.button("Explore Artworks", use_container_width=True):
            st.session_state.current_page = "explore"
    with head3:
        if st.button("New Project", use_container_width=True):
            st.session_state.current_page = "new"
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.current_page == "explore":
        st.markdown("### 🖼️ Explore Artworks Gallery")
        st.info("Your previously generated emotional masterpieces are showcased here.")
        
        outputs_dir = "outputs"
        if os.path.exists(outputs_dir):
            images = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) if f.endswith(('.png', '.jpg'))]
            # Sort newest first
            images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            if not images:
                st.warning("No artworks generated yet. Go to New Project to create one!")
            else:
                # Create a dynamic grid
                cols = st.columns(3)
                for idx, img_path in enumerate(images):
                    with cols[idx % 3]:
                        fname = os.path.basename(img_path)
                        # Extract date from filename if possible (e.g. emotion_art_20260710_225356)
                        caption_text = fname.replace("emotion_art_", "").replace(".png", "")
                        st.image(img_path, caption=f"Artwork: {caption_text}", use_container_width=True)
        else:
            st.warning("No artworks generated yet.")
        return

    
    # Split screen to match the Mockup (Left Sidebar settings, Right Main Canvas)
    left_col, right_col = st.columns([1, 3.5], gap="large")
    
    # Variables that need to be scoped globally for the run
    input_mode = "text"
    user_text = ""
    uploaded_file = None
    
    with left_col:
        st.markdown('<div class="dashboard-header">Generation Settings</div>', unsafe_allow_html=True)
        
        # Style selector
        style_choice = st.selectbox(
            "Artistic Style Template",
            options=["fused", "van_gogh", "monet", "da_vinci", "expressionism"],
            format_func=lambda x: {
                "fused": "Fused Masters (Van Gogh + Monet + Da Vinci)",
                "van_gogh": "Vincent van Gogh",
                "monet": "Claude Monet",
                "da_vinci": "Leonardo da Vinci",
                "expressionism": "Expressionism"
            }.get(x, x),
            index=0
        )
        
        steps = st.slider("Inference Steps", 5, 50, 25)
        guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
        
        st.selectbox("Image Size", ["16:9", "1:1", "4:3", "9:16"], index=0)
        st.selectbox("Sampler", ["DPM++ 2M Karras", "Euler A", "DDIM"], index=0)
        seed_in = st.text_input("Manual Seed (optional)", "")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-header">RTX & Optimizations</div>', unsafe_allow_html=True)
        
        # GPU low memory optimizations
        enable_cpu_offload = st.checkbox(
            "Enable Model CPU Offload",
            value=False,
            help="Saves substantial VRAM by unloading sub-modules from GPU."
        )
        
        show_heat = st.checkbox("Saturation Emphasis Overlay", value=True)
        show_faces = st.checkbox("Face Highlights", value=True)
        show_wc = st.checkbox("Explanation Word Cloud", value=True)
        
        st.caption(f"Engine: `{get_sd_model_id()}`")

    with right_col:
        st.markdown('<div class="dashboard-header" style="color:#8B5CF6">AI Prompt</div>', unsafe_allow_html=True)
        
        # We use tabs to switch between Text and Image, keeping it clean
        in_tab1, in_tab2 = st.tabs(["📝 Text Prompt", "📷 Image Upload (Facial Analysis)"])
        
        with in_tab1:
            user_text = st.text_area(
                "AI Prompt",
                value="I feel quiet and a bit lonely watching the rain fall outside.",
                placeholder="How do you feel?",
                key="text_input_area",
                label_visibility="collapsed"
            )
        
        with in_tab2:
            uploaded_file = st.file_uploader(
                "Upload a portrait or selfie for facial expression analysis",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                input_mode = "image"
                st.image(uploaded_file, caption="Target Image", width=200)

        # TRIGGER GENERATION
        submit_pressed = st.button("Generate Emotional Artwork", type="primary", use_container_width=True)
        
        if submit_pressed:
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
            status_box = st.status("Initializing Aurora AI Pipeline...", expanded=True)
            
            # Load local models
            status_box.write("Loading Transformers Detectors...")
            text_detector = load_text_detector()
            facial_detector = load_facial_detector()
            image_captioner = load_image_captioner()
            
            detected_emotion = "neutral"
            emotion_scores = {"neutral": 1.0}
            visual_caption = ""
            
            # Pipeline: Step 1 & 2 - Emotion Detection & Context analysis
            if input_mode == "text":
                status_box.write("Running Cognitive Text Analysis...")
                analysis_res = text_detector.detect(user_text)
                detected_emotion = analysis_res["emotion"]
                emotion_scores = analysis_res["scores"]
                visual_caption = f"Conceptual visualization of: {user_text}"
            else:
                status_box.write("Running Vision Scene & Facial Analysis...")
                pil_image = Image.open(uploaded_file).convert("RGB")
                face_res = facial_detector.detect(pil_image)
                detected_emotion = face_res["emotion"]
                emotion_scores = face_res["scores"]
                visual_caption = image_captioner.caption(pil_image)
                
            status_box.write(f"Detected dominant emotion: **{detected_emotion.upper()}**")
                
            # Pipeline: Step 3 - Prompt Engineering
            status_box.write("Engineering artistic prompt...")
            prompt_eng = PromptEngineer(default_style=style_choice)
            engineered_prompt = prompt_eng.engineer_prompt(
                emotion=detected_emotion,
                context_caption=visual_caption if input_mode == "image" else user_text,
                style_choice=style_choice
            )
            
            # Pipeline: Step 4 - Stable Diffusion Generation
            status_box.write("Synthesizing Masterpiece...")
            try:
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
            status_box.write("Extracting dominant color palette...")
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
            status_box.update(label="Artwork Generation Complete!", state="complete", expanded=False)
            
            # --- UI DISPLAY OF RESULTS ---
            st.markdown('<br><hr style="border-color:rgba(255,255,255,0.1)">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-header" style="font-size:1.5rem; color:#fff;">Generated Masterpiece</div>', unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                st.image(generated_img, caption="AI Synthesis Output", use_container_width=True)
                
                st.markdown('<div class="dashboard-header" style="margin-top:20px;">Color Palette</div>', unsafe_allow_html=True)
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
                
            with res_col2:
                st.metric(label="Primary Emotion", value=detected_emotion.upper())
                
                st.markdown('<div class="dashboard-header" style="margin-top:20px;">Confidence Scores</div>', unsafe_allow_html=True)
                sorted_scores = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(sorted_scores)
                
                with st.expander("Engineered Stable Diffusion Prompt", expanded=True):
                    st.code(engineered_prompt, language="text")
            
            st.markdown(f'<div class="critique-card">{explanation_md}</div>', unsafe_allow_html=True)
            
            payload = {
                "detected_emotion": detected_emotion,
                "visual_context": visual_caption,
                "engineered_prompt": engineered_prompt,
                "color_palette": [{"hex": c["hex"], "rgb": c["rgb"], "percentage": c["percentage"]} for c in color_palette],
                "saved_image_path": saved_path
            }
            dl_json = json.dumps(payload, indent=2, ensure_ascii=False)
            st.download_button("💾 Download Art Analysis JSON", data=dl_json, file_name="aurora_ai_result.json", use_container_width=True)
            
            # Analytics
            st.markdown('<br><hr style="border-color:rgba(255,255,255,0.1)">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-header" style="font-size:1.5rem; color:#fff;">Vision Analytics</div>', unsafe_allow_html=True)
            
            tabs = []
            if show_heat: tabs.append("Saturation Heatmap")
            if show_faces: tabs.append("Facial Boundaries")
            if show_wc: tabs.append("Keyword Cloud")
                
            if tabs:
                ts = st.tabs(tabs)
                ti = 0
                
                if show_heat:
                    with ts[ti]:
                        st.image(color_emphasis_overlay(generated_img), use_container_width=True)
                    ti += 1
                if show_faces:
                    with ts[ti]:
                        target_for_faces = Image.open(uploaded_file).convert("RGB") if input_mode == "image" else generated_img
                        face_img, n = highlight_faces(target_for_faces)
                        if n == 0: st.info("No faces detected.")
                        st.image(face_img, use_container_width=True)
                    ti += 1
                if show_wc:
                    with ts[ti]:
                        wc = word_cloud_image(analysis_text_for_wordcloud({"explanation": explanation_md}))
                        if wc is not None:
                            st.image(wc, use_container_width=True)
                        else:
                            st.info("Wordcloud disabled.")
                    ti += 1


if __name__ == "__main__":
    main()
