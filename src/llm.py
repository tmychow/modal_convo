"""
LLM with Mistral
"""

import time

from modal import Image, build, enter, method, gpu, Secret

from .common import stub

import os

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )
    move_cache()

mistral_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("torch", "transformers")
)

with mistral_image.imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer

@stub.cls(image=mistral_image, gpu=gpu.A100(memory=80, count=2), container_idle_timeout=300, concurrency_limit=8, allow_concurrent_inputs=True)
class Mistral:
    @build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_NAME)

    @enter()
    def load_model(self):
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        print(f"Model loaded in {time.time() - t0:.2f} seconds")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    @method()
    def generate(self, input, history=[]):
        t0 = time.time()
        assert len(history) % 2 == 0, "History must be an even number of messages"
        messages = []
        for i in range(0, len(history), 2):
            messages.append({"role": "user", "content": history[i]})
            messages.append({"role": "assistant", "content": history[i + 1]})
        messages.append({"role": "user", "content": input})

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)
        generated = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated)
        latest = decoded[0].split("[/INST]")[-1] if "[/INST]" in decoded[0] else decoded[0]
        stripped = latest.replace("</s>", "").strip()
        print(f"Response generated in {time.time() - t0:.2f} seconds")
        return stripped
    
# @stub.local_entrypoint()
# def main(input):
#     model = Mistral()
#     for val in model.generate.remote(input):
#         print(val, end="")