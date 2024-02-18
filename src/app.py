"""
Serves the API routes.
"""

from pathlib import Path

import os

import time

from modal import Mount, asgi_app

from .common import stub, vol
from .llm import Mistral
from .stt import Whisper
from .tts import XTTS

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import smtplib
from email.mime.text import MIMEText

def write_lora_config_to_volume(config_value):
    # Define the path to the file within the shared volume
    file_path = "/my-vol/lora_config.txt"
    # Write the config value to the file
    with open(file_path, "w") as file:
        file.write(config_value)
    vol.commit()

@stub.function(volumes={"/my-vol": vol})
@asgi_app()
def web():
    web_app = FastAPI()
    stt = Whisper()
    llm = Mistral()
    tts = XTTS()

    @web_app.get("/")
    def root():
        return {"message": "Hello, World"}
    
    # @web_app.post("/holding")
    # async def holding_respond(request: Request):
    #     body = await request.json()
    #     prompt = body["text"]
    #     tts.speak.remote(prompt)
    #     vol.reload()
    #     return FileResponse(path="/my-vol/output.wav", media_type="audio/x-wav")

    @web_app.post("/voice_response")
    async def voice_respond(request: Request, audio: UploadFile = File(...)):
        t0 = time.time()
        save_path = os.path.join("/my-vol/", audio.filename)
        with open(save_path, "wb") as buffer:
            buffer.write(await audio.read())
        vol.commit()
        # if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        #     print("File saved")
        transcript = stt.transcribe.remote(save_path)
        response = llm.generate.remote(transcript)
        tts.speak.remote(response)
        vol.reload()
        # if os.path.exists("/my-vol/output.m4a") and os.path.getsize("/my-vol/output.m4a") > 0:
        #     print("File saved")
        print(f"Pipeline done in {time.time() - t0:.2f} seconds")
        return FileResponse(path="/my-vol/output.wav", media_type="audio/x-wav")
    
    @web_app.post("/text_response")
    async def text_respond(request: Request):
        body = await request.json()
        prompt = body["text"]
        if "family" in prompt:
            write_lora_config_to_volume("1")
        elif "weightlifting" in prompt:
            write_lora_config_to_volume("2")
        response = llm.generate.remote(prompt)
        write_lora_config_to_volume("0")
        return JSONResponse(content={"text": response}, media_type="application/json")
        # return Response(content=response, media_type="text/plain")
        # return Response(content=response, media_type="text/plain")

    @web_app.post("/calendar_response")
    async def cal_respond(request: Request):
        body = await request.json()
        prompt = body["text"]
        if "family" in prompt:
            write_lora_config_to_volume("1")
        elif "weightlifting" in prompt:
            write_lora_config_to_volume("2")
        response = llm.generate.remote(prompt)
        write_lora_config_to_volume("0")

        subject = "Inspira Affirmation"
        body = response
        sender = "affirmations.ai.daily@gmail.com"
        recipients = ["sherylhsu02@gmail.com", "6692166410@mypixmessages.com"]
        password = "kyxm cvqz btun gnzx"


        def send_email(subject, body, sender, recipients, password):
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
                smtp_server.login(sender, password)
                smtp_server.sendmail(sender, recipients, msg.as_string())
            print("Message sent!")

        send_email(subject, body, sender, recipients, password)
    # web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app