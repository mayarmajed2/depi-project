"""Local AI Explanation System for generated emotional artwork.

Generates structured, art-critic-style explanations of why the artwork
looks the way it does, incorporating the detected emotion, style choices,
and actual dominant colors extracted from the generated image.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


def classify_rgb_color(r: int, g: int, b: int) -> str:
    """Classify a color as warm, cool, or neutral."""
    # Warm: high red/green, low blue
    # Cool: high blue, low red
    # Neutral: low saturation (R, G, B are close)
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    if diff < 25: # Low saturation -> neutral/grayish
        if max_val > 200:
            return "bright neutral (white/cream)"
        elif max_val < 50:
            return "dark neutral (black/shadow)"
        else:
            return "muted neutral (grey/taupe)"
            
    # Check hue dominance
    if r > g and r > b:
        if g > b * 1.5:
            return "warm golden-orange"
        return "warm fiery-red"
    elif b > r and b > g:
        if g > r * 1.2:
            return "cool teal/cyan"
        return "cool deep-blue"
    elif g > r and g > b:
        return "cool organic-green"
    elif r > b and g < b:
        return "warm/rich magenta-purple"
    return "balanced earthy tone"


class ArtExplainer:
    """Generates detailed artistic critiques explaining emotional artwork parameters."""

    def __init__(self):
        pass

    def generate_explanation(
        self,
        emotion: str,
        dominant_colors: List[Dict[str, Any]],
        style_choice: str = "fused",
        caption_context: Optional[str] = None
    ) -> str:
        """Create a beautiful markdown-formatted art critique explaining the image.
        
        Args:
            emotion: The detected emotion (e.g. 'sad', 'happy').
            dominant_colors: List of color dicts (rgb, hex, percentage) from color_analyzer.
            style_choice: The visual style template used.
            caption_context: Visual context/caption from BLIP.
            
        Returns:
            A multi-paragraph critique string.
        """
        emo = emotion.lower().strip()
        
        # --- Section 1: Mood & Emotional Expression ---
        mood_paras = {
            "happy": (
                f"The generated artwork vibrates with the essence of **{emotion}**. By translating "
                "this emotion into a visual language, the model establishes a sense of joy and optimism. "
                "The composition is open and uplifting, using forms that pull the viewer's eye upward "
                "to simulate the sensation of emotional lightness and freedom."
            ),
            "sad": (
                f"The artwork is built upon the visual representation of **{emotion}**. To capture this "
                "melancholic affect, the model structures a heavy, introspective space. Descending soft diagonals "
                "and empty spatial layouts are employed to represent quiet solitude, longing, and the slow, "
                "reflective tempo of sadness."
            ),
            "angry": (
                f"To depict the turbulent emotion of **{emotion}**, the artwork adopts a highly aggressive, "
                "chaotic visual structure. The painting relies on sharp, jagged forms and colliding lines to "
                "simulate tension, internal conflict, and the explosive release of energy characteristic of anger."
            ),
            "fearful": (
                f"The artwork visually interprets the emotion of **{emotion}** by evoking mystery, suspense, and "
                "psychological tension. Using claustrophobic framing and deep, receding paths, the composition "
                "induces a sense of vulnerability and the unknown."
            ),
            "surprised": (
                f"Capturing the sudden sensation of **{emotion}**, the artwork is configured with an energetic, "
                "unbalanced composition. Radial patterns and sudden bursts of light simulate a moment of wonder, "
                "shock, and sudden illumination."
            ),
            "disgusted": (
                f"The artwork expresses **{emotion}** through asymmetrical, winding, and slightly distorted "
                "forms that create a feeling of unease and tension. The textured brushwork feels rough and heavy, "
                "visually echoing feelings of decay and rejection."
            ),
            "neutral": (
                f"Representing a **{emotion}** emotional state, the painting focuses entirely on balance, "
                "stillness, and peaceful tranquility. The composition is highly symmetrical, with clean structural "
                "lines that invite a calm, meditative focus."
            )
        }
        
        # Fallback mood paragraph
        mood_text = mood_paras.get(
            emo,
            f"The artwork is designed around the emotional theme of **{emotion}**. The composition, brushwork, "
            "and structural balance are directly shaped to represent the unique psychological qualities of this mood."
        )
        
        # --- Section 2: Color Palette Analysis (Using actual extracted colors!) ---
        color_details = []
        color_types = []
        
        for idx, col in enumerate(dominant_colors[:3]): # Analyze top 3 colors
            r, g, b = col["rgb"]
            pct = col["percentage"] * 100
            hex_code = col["hex"]
            classification = classify_rgb_color(r, g, b)
            color_types.append(classification)
            color_details.append(f"`{hex_code}` ({classification}, accounting for {pct:.1f}% of the canvas)")
            
        color_intro = f"An analysis of the generated canvas reveals a dominant color palette featuring " + ", ".join(color_details) + "."
        
        color_meanings = {
            "happy": (
                " These bright, warm, and radiant hues are chosen specifically to trigger a psychological "
                "response of happiness and energy, reflecting the sunny side of human consciousness and vitality."
            ),
            "sad": (
                " These cool, desaturated, and dark tones are classic artistic symbols of melancholia. By lowering "
                "color saturation, the artwork evokes a quiet, submissive mood that allows sadness to take center stage."
            ),
            "angry": (
                " The presence of intense, fiery warm tones combined with dark neutrals visually communicates friction, "
                "heat, and agitation, mirroring the physiological intensity of anger and emotional conflict."
            ),
            "fearful": (
                " The murky greens, purples, and deep shadow tones create a chilling, desaturated atmosphere. These colors "
                "are selected to emulate the obscurity of night, casting shadows that suggest hidden dangers and mystery."
            ),
            "surprised": (
                " The contrasting mixture of bright electric colors against deep shadows represents a sudden spark "
                "of light breaking through dark space, symbolizing sudden awareness and revelation."
            ),
            "disgusted": (
                " The muddy, acidic tones are chosen to evoke feelings of decay and organic stagnation, using "
                "unappealing color relationships to reinforce the theme of aversion."
            ),
            "neutral": (
                " The soft, earthy, and highly balanced neutral colors foster a sense of grounding and calm, avoiding "
                "high-intensity colors to keep the emotional temperature perfectly balanced and peaceful."
            )
        }
        
        color_text = color_intro + color_meanings.get(emo, " These colors work in harmony to establish the overall emotional temperature of the narrative.")
        
        # --- Section 3: Lighting, Style & Composition ---
        lighting_style_text = ""
        if emo == "happy":
            lighting_style_text = (
                "The lighting is high-key and diffused, simulating sun-drenched outdoor illumination. The style avoids "
                "heavy shadows, creating a luminous, airy atmosphere that feels breathable and welcoming."
            )
        elif emo == "sad":
            lighting_style_text = (
                "The illumination is low-key, casting soft, melancholic shadows. The model uses dim overcast lighting "
                "with low contrast, creating a heavy atmosphere that restricts visual intensity to match the low energy of sadness."
            )
        elif emo == "angry":
            lighting_style_text = (
                "The scene features harsh, high-contrast chiaroscuro lighting, with bright hotspots directly clashing with "
                "deep, angular shadows. This dramatic lighting style creates a tense, theatrical atmosphere filled with friction."
            )
        elif emo == "fearful":
            lighting_style_text = (
                "The artwork is illuminated with eerie under-lighting and high-contrast tenebrism. Long, sweeping shadows "
                "stretch across the canvas, leaving large portions of the scene obscured to foster fear of the unseen."
            )
        elif emo in ["surprised", "surprised"]:
            lighting_style_text = (
                "The lighting resembles a sudden flash or halo of radial illumination, creating a glowing focus "
                "that highlights the central elements and represents a sudden, unexpected visual shock."
            )
        else:
            lighting_style_text = (
                "The lighting is soft, natural, and evenly balanced across the canvas, avoiding dramatic shadows or "
                "intense highlights to preserve a calm, harmonious composition."
            )

        # Integrate caption context if available
        if caption_context:
            context_phrase = f" The visual subject matter, centered around **{caption_context}**, is integrated into this spatial layout to tie the literal narrative to the emotional mood."
            lighting_style_text += context_phrase

        # --- Section 4: Artistic Techniques & Master Influences ---
        style_techs = {
            "fused": (
                "To manifest this vision, the system combines techniques from three legendary art historical masters: "
                "it uses the thick, swirling **impasto brushwork of Vincent van Gogh** to inject raw emotional energy, "
                "blends in the **soft, loose atmospheric brushwork and dappled light of Claude Monet** to build depth, "
                "and uses **Leonardo da Vinci's classical sfumato and chiaroscuro** to smooth color transitions and balance the forms."
            ),
            "van_gogh": (
                "The artwork is heavily inspired by Vincent van Gogh. It utilizes his signature **heavy impasto texture** "
                "and thick, energetic, swirling brushstrokes. The colors are applied with emotional urgency rather than "
                "photographic accuracy, creating expressive halos and dynamic surface textures that pulse with energy."
            ),
            "monet": (
                "The canvas pays tribute to Claude Monet's impressionist style. The brushwork is **soft, rapid, and loose**, "
                "prioritizing the fleeting effects of light and atmosphere over rigid details. Dappled light and soft color reflections "
                "merge to create a highly atmospheric, dreamlike quality."
            ),
            "da_vinci": (
                "The system references the classical techniques of Leonardo da Vinci. It employs **sfumato** to softly blend "
                "the edges and transitions of colors, removing harsh outlines. A delicate **chiaroscuro** is used to sculpt "
                "three-dimensional volumes with subtle light and shadow, resulting in a balanced, harmonious composition."
            ),
            "expressionism": (
                "The artwork uses raw **Expressionist techniques**, utilizing thick, aggressive palette knife strokes "
                "and distorted forms. Color is treated as a direct expression of the internal psyche, creating a highly tactile, "
                "textured surface that forces a visceral emotional connection with the viewer."
            )
        }
        
        tech_text = style_techs.get(style_choice, style_techs["fused"])
        
        # --- Build Final Markdown Output ---
        explanation_md = f"""### 🎨 AI Art Critique & Explanation

**Mood & Affect:**
{mood_text}

**Color Palette Theory:**
{color_text}

**Lighting & Composition:**
{lighting_style_text}

**Artistic Techniques:**
{tech_text}
"""
        return explanation_md
