"""
Text to speech with XTTS
"""

# TODO: Add checkpoint local so it doesn't reload
# Can't

import io
import tempfile

from pathlib import Path

from modal import Image, method

from .common import stub, vol


def download_models():
    from tortoise.api import MODELS_DIR, TextToSpeech

    tts = TextToSpeech(models_dir=MODELS_DIR)
    tts.get_random_conditioning_latents()


tortoise_image = (
    Image.debian_slim(python_version="3.10.8")  # , requirements_path=req)
    .apt_install("git", "libsndfile-dev", "ffmpeg", "curl")
    .pip_install(
        "torch==2.0.0",
        "torchvision==0.15.1",
        "torchaudio==2.0.1",
        "pydub==0.25.1",
        "transformers==4.25.1",
        extra_index_url="https://download.pytorch.org/whl/cu117",
    )
    .pip_install("git+https://github.com/metavoicexyz/tortoise-tts")
    .run_function(download_models)
)

@stub.cls(
    image=tortoise_image,
    gpu="A100",
    container_idle_timeout=300,
    timeout=180,
    volumes={"/my-vol": vol}
)
class XTTS:
    def __enter__(self):
        """
        Load the model weights into GPU memory when the container starts.
        """
        from tortoise.api import MODELS_DIR, TextToSpeech
        from tortoise.utils.audio import load_audio, load_voices

        self.load_voices = load_voices
        self.load_audio = load_audio
        self.tts = TextToSpeech(models_dir=MODELS_DIR)
        self.tts.get_random_conditioning_latents()

    def process_synthesis_result(self, result, output_path):
        """
        Converts a audio torch tensor to a binary blob.
        """
        import pydub
        import torchaudio

        torchaudio.save(
            output_path,
            result,
            24000,
        )

    @method()
    def speak(self, text, voices=["geralt"], output_path="/my-vol/output.wav"):
        text = text.strip()
        if not text:
            return

        CANDIDATES = 1  # NOTE: this code only works for one candidate.
        CVVP_AMOUNT = 0.0
        SEED = None
        PRESET = "fast"

        voice_samples, conditioning_latents = self.load_voices(voices)

        gen, _ = self.tts.tts_with_preset(
            text,
            k=CANDIDATES,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=PRESET,
            use_deterministic_seed=SEED,
            return_deterministic_state=True,
            cvvp_amount=CVVP_AMOUNT,
        )

        speaker_folder = Path(__file__).parent.absolute()
        self.process_synthesis_result(gen.squeeze(0).cpu(), output_path)
        vol.commit()
    



# import os

# from pathlib import Path

# os.environ["COQUI_TOS_AGREED"] = "1"

# from modal import Image, method, Mount

# from .common import stub, vol

# xtts_image = (
#     Image.debian_slim(python_version="3.10")
#     .pip_install("TTS", "deepspeed")
# )


# with xtts_image.imports():
#     from TTS.api import TTS

# # @stub.cls(image=xtts_image, gpu="A100", container_idle_timeout=180, mounts=[Mount.from_local_file(local_path="female.wav", remote_path="/root/src/female.wav"), Mount.from_local_dir(local_path="hold/XTTS-v2", remote_path="/root/src/XTTS-v2")], volumes={"/my-vol": vol})
# @stub.cls(image=xtts_image, gpu="A100", container_idle_timeout=180, mounts=[Mount.from_local_file(local_path="female.wav", remote_path="/root/src/female.wav")], volumes={"/my-vol": vol})
# class XTTS:
#     def __enter__(self):
#         self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
#         # from TTS.tts.configs.xtts_config import XttsConfig
#         # from TTS.tts.models.xtts import Xtts
#         # config = XttsConfig()
#         # config.load_json("/root/src/XTTS-v2/config.json")
#         # self.model = Xtts.init_from_config(config)
#         # self.model.load_checkpoint(config, checkpoint_dir="/root/src/XTTS-v2/checkpoints")
#         # self.model.cuda()

#     @method()
#     def speak(self, text, output_path="/my-vol/output.wav"):
#         speaker_folder = Path(__file__).parent.absolute()
#         self.tts.tts_to_file(text, speaker_wav=f"{speaker_folder}/female.wav", language="en", file_path=output_path)
#         vol.commit()
#         # output = self.model.synthesize(text, self.model.config, speaker_wav=f"{speaker_folder}/female.wav", language="en")
#         # with open(output_path, "wb") as file:
#         #     file.write(output)
#         # vol.commit()

# # @stub.local_entrypoint()
# # def main(input):
# #     model = XTTS()
# #     model.speak.remote(input)
# #     print("Audio file generated")