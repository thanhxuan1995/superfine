# %%
import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline
from PIL import Image
from moviepy import (
    ImageClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
)

# === STEP 1: Generate Ad Script / Hook Copy ===
text_generator = pipeline("text-generation", model="gpt2")

channel = "TikTok"
audience = "Gen Z Female"
product = "Fantasy RPG Game"

prompt = (
    f"Write a short catchy ad hook for a {product} targeting {audience} on {channel}."
)
script = text_generator(prompt, max_length=50, num_return_sequences=1)[0][
    "generated_text"
]
print("Generated Script:\n", script)

# %%
# AI-Powered Creative Generation Pipeline (Prototype)
# Requirements: transformers, diffusers, PIL, moviepy, torch, openai, huggingface_hub

import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline
from PIL import Image
from moviepy import (
    ImageClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
)

# === STEP 1: Generate Ad Script / Hook Copy ===
text_generator = pipeline("text-generation", model="gpt2")

channel = "TikTok"
audience = "Gen Z Female"
product = "Fantasy RPG Game"

prompt = (
    f"Write a short catchy ad hook for a {product} targeting {audience} on {channel}."
)
script = text_generator(prompt, max_length=50, num_return_sequences=1)[0][
    "generated_text"
]
print("Generated Script:\n", script)

# === STEP 2: Generate Visual Theme Using Stable Diffusion ===
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
sd_pipe = sd_pipe.to("cuda")

image_prompt = (
    f"An epic fantasy battle scene with a strong female hero, trending on {channel}"
)
image = sd_pipe(image_prompt).images[0]
image.save("frame_1.png")

# === STEP 3: Turn Script + Visuals into Video Creative ===
clip_duration = 5  # seconds

image_clip = ImageClip("frame_1.png").set_duration(clip_duration).resize(height=1080)

text_overlay = TextClip(
    script,
    fontsize=50,
    color="white",
    font="Arial-Bold",
    method="caption",
    size=(1080, None),
)
text_overlay = text_overlay.set_position(("center", "bottom")).set_duration(
    clip_duration
)

final_clip = CompositeVideoClip([image_clip, text_overlay])
final_clip.write_videofile("ad_creative_tiktok.mp4", fps=24)

# === STEP 4: (Optional) Use Voiceover or Audio ===
# You can use ElevenLabs API or pyttsx3 to generate a voiceover from the script.

# === STEP 5: Repeat / Adapt for Facebook, YouTube, etc ===
# Change aspect ratio, hook tone, or video length depending on platform.

print("✅ Creative set generated successfully!")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
import tensorflow

# %%
!pipenv install tensorflow

# %%


# %%


# %%


# %%



