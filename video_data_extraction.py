# %%
import cv2
import ffmpeg


def extract_basic_info(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    return {
        "Total frames:": frame_count,
        "FPS:": fps,
        "Resolution_w": width,
        "Resolution_h": height,
        "Duration": duration,
    }


basic_info = extract_basic_info("input.mp4")
basic_info

# %%
import cv2
import numpy as np


def extract_optical_flow(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback optical flow (dense)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return {"motion": np.array(flows)}  # shape: (num_frames - 1, H, W, 2)


# Example usage
flow_data = extract_optical_flow("input.mp4")
flow_data["motion"].shape

# %%


# %%
def extract_video_metadata(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe["format"]
        video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        subtitle_streams = [
            s for s in probe["streams"] if s["codec_type"] == "subtitle"
        ]

        metadata = {
            "filename": format_info.get("filename"),
            "duration": float(format_info.get("duration", 0)),
            "bit_rate": int(format_info.get("bit_rate", 0)),
            "video_codec": video_streams[0]["codec_name"] if video_streams else None,
            "width": int(video_streams[0]["width"]) if video_streams else None,
            "height": int(video_streams[0]["height"]) if video_streams else None,
            "fps": eval(video_streams[0]["avg_frame_rate"]) if video_streams else None,
            "audio_codec": audio_streams[0]["codec_name"] if audio_streams else None,
            "subtitles": len(subtitle_streams),
        }

        return metadata
    except ffmpeg.Error as e:
        print("Error extracting metadata:", e)
        return {}


# Example usage
metadata = extract_video_metadata("input.mp4")
for k, v in metadata.items():
    print(f"{k}: {v}")

# %%
metadata

# %%
basic_info

# %%
import cv2
import ffmpeg
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


class VideoFeatureExtraction:
    def __init__(self, video_path):
        self.video_path = video_path

    def extract_basic_info(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of rame.
            fps = cap.get(cv2.CAP_PROP_FPS)  # number of frame per second.
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps
            return {
                "Total frames:": frame_count,
                "FPS:": fps,
                "Resolution_w": width,
                "Resolution_h": height,
                "Duration": duration,
            }
        except Exception as e:
            logging("Error in extract basic infomation is:", e)
            return {}

    def extract_video_metadata(self):
        try:
            probe = ffmpeg.probe(self.video_path)
            format_info = probe["format"]
            video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
            audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
            subtitle_streams = [
                s for s in probe["streams"] if s["codec_type"] == "subtitle"
            ]
            metadata = {
                "filename": format_info.get("filename"),
                "duration": float(format_info.get("duration", 0)),
                "bit_rate": int(format_info.get("bit_rate", 0)),
                "video_codec": (
                    video_streams[0]["codec_name"] if video_streams else None
                ),
                "width": int(video_streams[0]["width"]) if video_streams else None,
                "height": int(video_streams[0]["height"]) if video_streams else None,
                "fps": (
                    eval(video_streams[0]["avg_frame_rate"]) if video_streams else None
                ),
                "audio_codec": (
                    audio_streams[0]["codec_name"] if audio_streams else None
                ),
                "subtitles": len(subtitle_streams),
            }

            return metadata
        except ffmpeg.Error as e:
            print("Error extracting metadata:", e)
            return {}

    def call(self):
        return {**self.extract_basic_info(), **self.extract_video_metadata()}

# %%
## read file and prepare data
sql = """SELECT * FROM raw_network_edges
WHERE raw_network_edges.to_id like '%.mp4'
limit 10;"""

# %%
import pandas as pd

df = pd.read_csv("final_mp4_list.csv")

# %%
df.head()

# %%
df["to_id"][0]

# %%
VideoFeatureExtraction(df["to_id"][0]).call()

# %%
dff = df[:10]

# %%
dff["video_info"] = dff["to_id"].apply(lambda x: VideoFeatureExtraction(x).call())

# %%
dff["video_info"]

# %%
### create y target.
dff.head(1)

# %%
target_cols = ["revenue", "clicks", "installs", "impressions", "spends"]

# %%
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Assume dff['video_info'] is already dict-like
def flatten_video_info(row):
    return pd.Series(row["video_info"])


# Create numerical features
video_df = dff["video_info"].apply(pd.Series).fillna(0)
video_df

# %%
video_df.info()

# %%
# Select only numeric columns
video_df_numeric = video_df.select_dtypes(include=["number"])

# Select only object (textual) columns for LLM
video_df_text = video_df.select_dtypes(include=["object"])

# Optionally: combine all object fields into one string per row
video_df_text["combined_text"] = video_df_text.apply(
    lambda row: " ".join(row.dropna().astype(str)), axis=1
)
# Final text input
text_input_list = video_df_text["combined_text"].tolist()

# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_text = tokenizer(
    text_input_list, return_tensors="pt", padding=True, truncation=True, max_length=128
)

# %%
tokenized_text["input_ids"].shape

# %%
## convert numeric columns to tensor
import torch

numerical_input = torch.tensor(video_df_numeric.values, dtype=torch.float32)

# %%
numerical_input.shape

# %%
## build model
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# %%
target_cols

# %%


# %%
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from io import BytesIO
import requests
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Download video
video_url = "https://res1.applovin.com/o5fc8647/36e5757c13bd9647f476212ac502707471a13431_v23_phone.mp4"
video_data = requests.get(video_url).content
video_path = "/tmp/video.mp4"
with open(video_path, "wb") as f:
    f.write(video_data)


# Extract frames from video
def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


frames = extract_frames(video_path)

# Generate captions for each frame
captions = []
for frame in frames:
    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)
    # Preprocess and generate caption
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append(caption)

# Combine captions to form a description
video_description = " ".join(captions)
print("Generated Video Description:")
print(video_description)

# %%
import torch

print(torch.__version__)

# %%
import numpy

print(numpy.__version__)

# %%
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from io import BytesIO
import requests
from PIL import Image
import ffmpeg
import logging

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Download video
video_url = "https://res1.applovin.com/o5fc8647/36e5757c13bd9647f476212ac502707471a13431_v23_phone.mp4"
video_data = requests.get(video_url).content
video_path = "/tmp/video.mp4"
with open(video_path, "wb") as f:
    f.write(video_data)


# Extract frames from video
def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


frames = extract_frames(video_path)

# Generate captions for each frame
captions = []
for frame in frames:
    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)
    # Preprocess and generate caption
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append(caption)

# Combine captions to form a description
video_description = " ".join(captions)


# Extract additional video information
def extract_video_info(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe["format"]
        video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        subtitle_streams = [
            s for s in probe["streams"] if s["codec_type"] == "subtitle"
        ]

        # Try to extract 'nb_frames' from the probe output, otherwise use OpenCV
        total_frames = None
        if "nb_frames" in format_info:
            total_frames = int(format_info["nb_frames"])
        else:
            # Fallback to OpenCV if 'nb_frames' is not available
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        # Gather information
        video_info = {
            "Total frames": total_frames,
            "FPS": video_streams[0]["r_frame_rate"] if video_streams else None,
            "Resolution_w": int(video_streams[0]["width"]) if video_streams else None,
            "Resolution_h": int(video_streams[0]["height"]) if video_streams else None,
            "Duration": float(format_info["duration"]),
            "filename": format_info["filename"],
            "bit_rate": (
                int(format_info["bit_rate"]) if "bit_rate" in format_info else None
            ),
            "video_codec": video_streams[0]["codec_name"] if video_streams else None,
            "width": int(video_streams[0]["width"]) if video_streams else None,
            "height": int(video_streams[0]["height"]) if video_streams else None,
            "fps": eval(video_streams[0]["avg_frame_rate"]) if video_streams else None,
            "audio_codec": audio_streams[0]["codec_name"] if audio_streams else None,
            "subtitles": len(subtitle_streams),
        }

        return video_info
    except ffmpeg.Error as e:
        logging.error("Error extracting video info: %s", e)
        return {}


# Get video information
video_info = extract_video_info(video_path)

# Format video info into a description string
video_info_description = "\n".join(
    [f"{key}: {value}" for key, value in video_info.items() if value is not None]
)

# Combine the generated captions and the video information
full_video_description = f"Video Information:\n{video_info_description}\n\nGenerated Video Description:\n{video_description}"

print("Full Video Description:")
print(full_video_description)

# %%
video_description

# %%
video_info["video_description"] = video_description

# %%
video_info

# %%
from transformers import pipeline

# %%
llm = pipeline(
    "text-generation", model="openai-gpt", device=0 if torch.cuda.is_available() else -1
)

# %%
def extract_tags_from_captions(captions):
    prompt = f"""Your are video game expert description.
        Your task is:
        Extract concise tags (2-3 words each) that describe 
        themes, characters, actions, or UI elements from this video game each frame description: {captions}
        only return the dictionary values like this  
        "theme": your value 2-3 words,
        "characters" : your observed character 2-3 words,
        "actions" : your observed action movement 2-3 words,
        "UI elements" : your observed in UI elements 2-3 words
        """
    result = llm(prompt, max_new_tokens=50, do_sample=True, top_k=20)[0][
        "generated_text"
    ]
    return result.split("Tags:")[-1].strip()

# %%
tags = extract_tags_from_captions(captions=video_description)

# %%
tags

# %%
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# %%
class OutputStructure(BaseModel):
    Gameplay_Mechanics: str = Field(
        description="Describe the core interactions the player has, example: Endless runner, Merge mechanic, Puzzle solving, Idle clicker, Tower defense, Drag-and-drop, First-person shooter, etc..."
    )
    Art_Style: str = Field(
        description="Describes the aesthetic of the game, example: Cartoon style, Realistic graphics, Pixel art, Low poly, 3D render, Dark theme, Neon lights, Warm palette, etc..."
    )
    Environment: str = Field(
        description="Describe the world or level types example: Cityscap, Dungeon, Kitchen theme, Outer space, Fantasy land, Road run, Underwater, etc..."
    )
    Character_Type: str = Field(
        description="Tag the main character’s nature, example: Human avatar, Animal protagonist, Cartoon character, Robot, Granny character, Superhero, etc..."
    )
    CTA: str = Field(
        description="Presence of direct or indirect user Call To Actions example: Play now, Tap to play, Download today, Before/after reveal, Reward effect, etc..."
    )
    Format: str = Field(
        description="Tag based on ad format and platform clues example: Mobile ad format, Playable ad, App Store badge, Landscape video, Portrait video, etc..."
    )
    Tone: str = Field(
        description="Capture the emotional appeal of the ad, example: Satisfying animation, Cozy environment, High-stakes challenge, Comedy, Fast-paced"
    )
    Reward_Systems: str = Field(
        description="Highlight features tied to progress or monetization example: Progression visual, Loot box, Customization, Unlockable content, Daily rewards"
    )

# %%
system_prompt = SystemMessagePromptTemplate.from_template(
    "System: You are an AI video game analyst. You will extract structured high-level tags from game video descriptions."
)
AI_prompt = AIMessagePromptTemplate.from_template(
    "Using the format below, extract one high-level summary per category.\n\n"
    "{format_instructions}\n\n"
    "Respond with only valid JSON, without any commentary, markdown, or explanations."
)
human_prompt = HumanMessagePromptTemplate.from_template(
    "User: Here's a textual description of the video content:\n\n{user_input}"
)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt, AI_prompt])

# %%


# %%
GROQ_API_KEY = "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"
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "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"
LANGSMITH_PROJECT = "agent"
based_prompt = "hwchase17/react"
model = "llama-3.3-70b-versatile"
temperature = 0.8
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=GROQ_API_KEY)

# %%
parser = JsonOutputParser(pydantic_object=OutputStructure)
format_instructions = parser.get_format_instructions()

# %%
chain = chat_prompt | llm | parser

# %%
res = chain.invoke(
    {"user_input": video_info, "format_instructions": format_instructions}
)

# %%
print(res)

# %%
res

# %%



