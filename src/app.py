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
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

class Prompt(BaseModel):
    text: str

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
    #     return Response(content=prompt, media_type="text/plain")
    #     body = await request.json()
    #     prompt = body["text"]
    #     response = llm.generate.remote(prompt)
    #     return Response(content=response, media_type="text/plain")

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
        return FileResponse(path="/my-vol/output.m4a")
    
    @web_app.post("/text_response")
    async def text_respond(request: Request):
        body = await request.json()
        prompt = body["text"]
        response = llm.generate.remote(prompt)
        return Response(content=response, media_type="text/plain")
    
    # web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app