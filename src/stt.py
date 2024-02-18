"""
Speech to text with Whisper
"""

from modal import Image, method

from .common import stub

MODEL_NAME = "openai/whisper-large-v3"

whisper_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("torch", "transformers")
)

with whisper_image.imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer