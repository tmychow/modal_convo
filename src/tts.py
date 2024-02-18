"""
Text to speech with XTTS
"""

import os

from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

from modal import Image, method, Mount

from .common import stub

xtts_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("TTS")
)

with xtts_image.imports():
    from TTS.api import TTS

@stub.cls(image=xtts_image, gpu="T4", container_idle_timeout=180, mounts=[Mount.from_local_file(local_path="female.wav", remote_path="/root/src/female.wav")])
class XTTS:
    def __enter__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    @method()
    def speak(self, text, output_path="output.wav"):
        speaker_folder = Path(__file__).parent.absolute()
        self.tts.tts_to_file(text, speaker_wav=f"{speaker_folder}/female.wav", language="en", file_path=output_path)

# @stub.local_entrypoint()
# def main(input):
#     model = XTTS()
#     model.speak.remote(input)
#     print("Audio file generated")