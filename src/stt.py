"""
Speech to text with Whisper
"""

import time

from modal import Image, method, Mount

from .common import stub

MODEL_NAME = "openai/whisper-large-v3"

whisper_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("torch", "transformers", "ffmpeg-python")
)

with whisper_image.imports():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@stub.cls(image=whisper_image, gpu="A10G", container_idle_timeout=180, mounts=[Mount.from_local_file(local_path="female.wav", remote_path="/root/src/female.wav")])
class Whisper:
    def __enter__(self):
        import torch

        self.use_cuda = torch.cuda.is_available()
        device = "cuda" if self.use_cuda else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to(device)

    @method()
    def transcribe(self, audio):
        t0 = time.time()
        pipe = pipeline(
        "automatic-speech-recognition",
        model=self.model,
        tokenizer=self.processor.tokenizer,
        feature_extractor=self.processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        )
        result = pipe(audio)
        print(f"Transcription completed in {time.time() - t0:.2f} seconds")
        print(result["text"])
        return result["text"]

# @stub.local_entrypoint()
# def main():
#     model = Whisper()
#     model.transcribe.remote("/root/src/female.wav")

