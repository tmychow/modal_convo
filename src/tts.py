"""
Text to speech with XTTS
"""

import os

from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

from modal import Image, method, Mount

from .common import stub, vol

xtts_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("TTS", "deepspeed")
)

with xtts_image.imports():
    from TTS.api import TTS

@stub.cls(image=xtts_image, gpu="A100", container_idle_timeout=180, mounts=[Mount.from_local_file(local_path="female.wav", remote_path="/root/src/female.wav")], volumes={"/my-vol": vol})
class XTTS:
    def __enter__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    @method()
    def speak(self, text, output_path="/my-vol/output.m4a"):
        speaker_folder = Path(__file__).parent.absolute()
        self.tts.tts_to_file(text, speaker_wav=f"{speaker_folder}/female.wav", language="en", file_path=output_path)
        vol.commit()

# @stub.local_entrypoint()
# def main(input):
#     model = XTTS()
#     model.speak.remote(input)
#     print("Audio file generated")