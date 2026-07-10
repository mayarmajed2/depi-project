"""Local Prompt Engineering module.

Bridges emotion analysis, image understanding captions, and artistic styles
to construct highly detailed prompts for local image generation.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

# Style mappings for emotions to guide prompt generation
EMOTION_STYLES = {
    "happy": {
        "colors": "vibrant warm colors, bright golden yellows, sunny oranges, and luminous pinks",
        "lighting": "sun-drenched morning light, bright soft highlighting, and luminous reflections",
        "composition": "open and energetic composition, flowing upward curves, high contrast, uplifting presence",
        "keywords": "joyful atmosphere, dancing light, positive energy, radiant beauty"
    },
    "sad": {
        "colors": "deep blues, cool indigo tones, melancholic slate greys, and muted teal shades",
        "lighting": "dim overcast twilight, soft chiaroscuro shadows, and low contrast moody illumination",
        "composition": "closed and introspective composition, descending soft diagonal lines, quiet heavy space, isolated subject",
        "keywords": "melancholic atmosphere, quiet solitude, weeping willow shadows, emotional calmness"
    },
    "angry": {
        "colors": "fiery crimson reds, stark charcoals, intense deep blacks, and blood orange accents",
        "lighting": "dramatic chiaroscuro lighting, harsh angular shadows, and extreme high-contrast illumination",
        "composition": "fragmented and explosive composition, sharp jagged intersecting lines, tense central focus",
        "keywords": "violent energy, tempestuous storm, raw tension, emotional friction, dynamic impact"
    },
    "fearful": {
        "colors": "desaturated olive greens, cold murky purples, shadow blacks, and ghostly pale yellow highlights",
        "lighting": "eerie under-lighting, long stretching shadows, high-contrast tenebrism, casting dark silhouettes",
        "composition": "claustrophobic and off-center composition, sharp diagonals, deep bottomless perspective, narrow paths",
        "keywords": "haunting atmosphere, mystery, hidden shadows, suspenseful depth, psychological tension"
    },
    "surprised": {
        "colors": "sudden pops of electric magenta, bright cyan, neon gold, set against contrasting deep violet",
        "lighting": "flash-like burst of light, radial highlighting, glowing ethereal halos, lens flares",
        "composition": "unbalanced and dynamic composition, radiating lines, centrifugal motion, sudden suspension",
        "keywords": "wonder, breathtaking shock, electric spark, sudden revelation, magic realism"
    },
    "disgusted": {
        "colors": "sickly greenish-yellows, swamp greens, muddy browns, and bruised purples",
        "lighting": "flat sickly fluorescent illumination, casting dirty green shadows, cold dim light",
        "composition": "asymmetrical off-putting composition, winding distorted forms, uneasy balance",
        "keywords": "decay, industrial grime, swamp mist, heavy air, raw expressionism"
    },
    "neutral": {
        "colors": "soft earthy ochre, muted beige, warm taupe, and quiet slate gray",
        "lighting": "flat diffused natural midday light, balanced clean illumination, soft shadows",
        "composition": "highly balanced and symmetrical classical composition, central horizontal focus",
        "keywords": "peaceful stillness, zen-like tranquility, balanced harmony, classical poise"
    }
}

# Base artistic style templates
ART_STYLE_TEMPLATES = {
    "fused": (
        "masterpiece oil painting combining post-impressionist impasto and swirling energetic "
        "brushwork with impressionist soft loose brushwork, dappled light and atmospheric depth, "
        "and Renaissance sfumato with subtle chiaroscuro and balanced classical composition"
    ),
    "van_gogh": (
        "post-impressionist masterpiece painting, thick impasto textures, swirling heavy brushstrokes, "
        "intense emotional color application, energetic star-like halos and expressive forms"
    ),
    "monet": (
        "impressionist masterpiece oil painting, soft loose brushwork, dappled light and reflections, "
        "atmospheric depth, delicate textures, capturing the fleeting effects of outdoor light"
    ),
    "da_vinci": (
        "Renaissance masterpiece painting, smooth sfumato technique, subtle chiaroscuro shading, "
        "balanced classical composition, highly detailed and realistic human figures, soft skin transitions"
    ),
    "expressionism": (
        "raw expressionist oil painting, distorted emotional figures, violent thick palette knife textures, "
        "exaggerated colors representing internal psychological state, high emotional impact"
    )
}

from emotion_art.prompts.constants import NEGATIVE_PROMPT


class PromptEngineer:
    """Class to construct emotional artistic prompts for Stable Diffusion."""

    def __init__(self, default_style: str = "fused"):
        self.default_style = default_style

    def engineer_prompt(
        self,
        emotion: str,
        context_caption: Optional[str] = None,
        style_choice: Optional[str] = None
    ) -> str:
        """Engineer an artistic prompt combining emotion details, caption, and artistic style.
        
        Args:
            emotion: Detected emotion (e.g. 'sad', 'happy').
            context_caption: Image caption/visual context (from BLIP or user).
            style_choice: Theme style (fused, van_gogh, monet, da_vinci, expressionism).
            
        Returns:
            The engineered prompt string.
        """
        # Resolve emotion styles
        emo = emotion.lower().strip()
        if emo not in EMOTION_STYLES:
            # Fallback to closest match or neutral
            matched_emo = "neutral"
            for k in EMOTION_STYLES:
                if k in emo:
                    matched_emo = k
                    break
            emo = matched_emo
            
        style_data = EMOTION_STYLES[emo]
        
        # Resolve base art style
        style_key = style_choice or self.default_style
        if style_key not in ART_STYLE_TEMPLATES:
            style_key = "fused"
        art_base = ART_STYLE_TEMPLATES[style_key]

        # Handle context caption
        # If BLIP caption is provided, we weave it into the prompt.
        # Example caption: "a woman sitting in a room"
        # We transform it into: "A beautiful oil painting of a woman sitting in a room..."
        subject = ""
        if context_caption:
            cap = context_caption.strip().lower()
            # Remove common starting phrases if present
            prefixes_to_strip = ["a painting of", "a photo of", "an image of", "a drawing of"]
            for prefix in prefixes_to_strip:
                if cap.startswith(prefix):
                    cap = cap[len(prefix):].strip()
            
            # Capitalize and set subject
            subject = f"representing {cap}, "
        else:
            subject = "depicting an emotional visual narrative, "

        # Assemble prompt pieces
        prompt_parts = [
            art_base,
            subject,
            f"conveying intense {emotion.upper()} mood and theme",
            f"characterized by {style_data['colors']}",
            f"illuminated with {style_data['lighting']}",
            f"featuring an {style_data['composition']}",
            f"infused with {style_data['keywords']}",
            "highly detailed, museum quality, 8k resolution"
        ]
        
        return ", ".join(prompt_parts)
