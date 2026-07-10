# 🎨 Emotion-to-Art AI — Local & Offline Multimodal AI System

An advanced, offline AI pipeline that analyzes human emotions from text descriptions or uploaded images/portraits, translates them into custom artistic styles using rule-based prompts, and generates beautiful, high-fidelity artworks locally.

Developed as a showcase for DEPI project.

---

## 🌟 Key Features

*   **Multimodal Emotion Detection**:
    *   **Text Analysis**: DistilRoBERTa model classifies text input into 7 distinct emotion scores.
    *   **Facial Recognition**: Vision Transformer (ViT) classifies emotions from faces in uploaded images.
    *   **Scene Captioning**: Salesforce BLIP models describe context and visual setup.
*   **Artistic Prompt Engineering**: Programmatic mapping of emotions to professional lighting, styles, color palettes, and negative prompts.
*   **Local Art Generation**: Generates high-fidelity artwork using Stable Diffusion (`runwayml/stable-diffusion-v1-5`) running completely locally.
*   **Computer Vision & Color Analytics**:
    *   Extracts the 5 most dominant color hex codes and percentages using OpenCV K-Means clustering.
    *   Applies a color saturation heat-map overlay to analyze visual weight.
    *   Uses Haar Cascade classifiers to outline faces in the uploaded/generated image.
*   **Art Critique Word Cloud**: Generates interactive word clouds based on the AI's artistic decision explanations.

---

## 🚀 Quick Start (One-Click Launch)

This project includes a pre-configured execution runner. You don't need to manually install dependencies or set up environments:

1.  Clone this repository or download the project files.
2.  Double-click the **`run.bat`** file in the root folder.
3.  The runner will automatically:
    *   Install and update all dependencies (`torch`, `torchvision`, `diffusers`, `transformers`, etc.).
    *   Launch the Streamlit web application.
    *   Open your browser directly to `http://localhost:8501`.

---

## 💻 Manual Setup

If you prefer to run the project manually from your terminal:

```powershell
# Navigate to the inner project directory
cd "depi project"

# Install all requirements
py -m pip install -r requirements.txt
py -m pip install torchvision

# Run the Streamlit application
py -m streamlit run app.py
```

---

## 🛠️ Offline Model Architecture

All models are downloaded and run locally in PyTorch on your machine (no external API calls):
1.  **Text Emotion Classifier**: `j-hartmann/emotion-english-distilroberta-base`
2.  **Facial Emotion Classifier**: `dima806/facial_emotions_image_detection`
3.  **Image Captioning (BLIP)**: `Salesforce/blip-image-captioning-base`
4.  **Stable Diffusion**: `runwayml/stable-diffusion-v1-5` (or custom ID in `.env`)
5.  **Analytics**: OpenCV, WordCloud, and Matplotlib.

---

## 🎨 Customizable Styles
You can select from multiple preset templates in the UI:
- **Fused Masters**: A blend of Van Gogh, Monet, and Da Vinci.
- **Vincent van Gogh**: High impasto, swirls, and expressive textures.
- **Claude Monet**: Soft, atmospheric Impressionism.
- **Leonardo da Vinci**: Sfumato, chiaroscuro, and classical drawings.
- **Expressionism**: Raw, exaggerated, and intense emotions.

---

## 📦 Directory Structure

```
depi project/
│
├── run.bat                     # One-click installer & launcher
├── README.md                   # Project documentation
│
└── depi project/               # Main source code folder
    ├── app.py                  # Streamlit Web UI Entrypoint
    ├── requirements.txt        # Package requirements
    ├── .env.example            # Environment variables configuration example
    ├── emotion_art/            # AI Pipeline Logic
    │   ├── analysis/           # Emotion Classifiers & BLIP Captioner
    │   ├── generation/         # Stable Diffusion Art Generation
    │   ├── prompts/            # Constants & Prompt Engineering
    │   └── utils/              # Color Analysis & Visualizations
    └── outputs/                # Folder where generated artwork is saved locally
```

---

## 📄 License & Credits
Developed by **Mayar** as part of the DEPI Project. Powered by Hugging Face, Streamlit, and PyTorch.
