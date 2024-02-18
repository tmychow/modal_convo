"""
LLM with Mistral
"""

# TODO: add LORA

import time

from modal import Image, build, enter, method, gpu, Secret, Mount

from .common import stub, vol

import os

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

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

# mistral_image = (
#     Image.debian_slim(python_version="3.10")
#     .pip_install("torch", "transformers")
# )

mistral_image = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm", "transformers", "huggingface_hub", "torch", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, secrets=[Secret.from_name("my-huggingface-secret")])
)

with mistral_image.imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer

@stub.cls(image=mistral_image, gpu="A100", secrets=[Secret.from_name("my-huggingface-secret")], mounts=[Mount.from_local_dir(local_path="lora", remote_path="/root/src/lora")], volumes={"/my-vol": vol})
class Mistral:
    def __enter__(self):
        vol.reload()
        def read_lora_config():
            with open("/my-vol/lora_config.txt", "r") as file:
                return file.read()
    
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.llm = LLM(MODEL_DIR, enable_lora=False)
        self.template = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] """

    # @build()
    # def download_model(self):
    #     from huggingface_hub import snapshot_download
    #     snapshot_download(MODEL_NAME)

    # @enter()
    # def load_model(self):
    #     t0 = time.time()
    #     self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    #     print(f"Model loaded in {time.time() - t0:.2f} seconds")
    #     self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    @method()
    def generate(self, input, history=[]):
        # t0 = time.time()
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        vol.reload()
        def read_lora_config():
            with open("/my-vol/lora_config.txt", "r") as file:
                return file.read()
        # assert len(history) % 2 == 0, "History must be an even number of messages"
        # messages = []
        # for i in range(0, len(history), 2):
            # messages.append({"role": "user", "content": history[i]})
            # messages.append({"role": "assistant", "content": history[i + 1]})
        # messages.append({"role": "user", "content": input})

        system_text = ""
        user_text = ""
        
        if "<SEP>" in input:
            system_text, user_text = input.split("<SEP>", 1)
            system_text = "You are a helpful friendly assistant who gives supportive pep talks! Here is some information about the user: " + system_text
        else:
            system_text = "You are a helpful friendly assistant who gives supportive pep talks!"
            user_text = input

        model_inputs = []
        if read_lora_config() == "1":
            model_inputs = [
                self.template.format(system=system_text + " Please use lots of emojis!", user=user_text)
            ]
        elif read_lora_config() == "2":
            model_inputs = [
                self.template.format(system=system_text + "CAPITALIZE EVERY LETTER IN EXCITEMENT!", user=user_text)
            ]
        else:
            model_inputs = [
                self.template.format(system=system_text, user=user_text)
            ]
        # if read_lora_config() == "1":
        #     model_inputs = [
        #         self.template.format(system=system_text + " Please use lots of emojis!", user=q) for q in [input]
        #     ]
        # elif read_lora_config() == "2":
        #     model_inputs = [
        #         self.template.format(system="You are a helpful friendly assistant who gives supportive pep talks! CAPITALIZE EVERY LETTER IN EXCITEMENT!", user=q) for q in [input]
        #     ]
        # else:
        #     model_inputs = [
        #         self.template.format(system="You are a helpful friendly assistant gives supportive pep talks!", user=q) for q in [input]
        #     ]
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=100,
            presence_penalty=1.15,
        )
        generation = None
        if read_lora_config() == "1":
            # generation = self.llm.generate(model_inputs, sampling_params=sampling_params, lora_request=LoRARequest("c3po-fam", 1, "/root/src/lora/family"))
            generation = self.llm.generate(model_inputs, sampling_params=sampling_params)
        elif read_lora_config() == "2":
            # generation = self.llm.generate(model_inputs, sampling_params=sampling_params, lora_request=LoRARequest("c3po-weight", 2, "/root/src/lora/weightlifting"))
            generation = self.llm.generate(model_inputs, sampling_params=sampling_params)
        else:
            generation = self.llm.generate(model_inputs, sampling_params=sampling_params)
        # decoded = self.llm.decode(result)
        # latest = decoded[0].split("[/INST]")[-1] if "[/INST]" in decoded[0] else decoded[0]
        # stripped = latest.replace("</s>", "").strip()
        result = ""
        for output in generation:
            result += output.outputs[0].text
        # if read_lora_config() == "2":
        #     return result.upper()
        # else:
        return result

        # encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # model_inputs = encodeds.to(self.model.device)
        # generated = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        # decoded = self.tokenizer.batch_decode(generated)
        # latest = decoded[0].split("[/INST]")[-1] if "[/INST]" in decoded[0] else decoded[0]
        # stripped = latest.replace("</s>", "").strip()
        # print(f"Response generated in {time.time() - t0:.2f} seconds")
        # return stripped
    
# @stub.local_entrypoint()
# def main(input):
#     model = Mistral()
#     for val in model.generate.remote(input):
#         print(val, end="")