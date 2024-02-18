"""
Serves the API routes.
"""

from pathlib import Path

from modal import Mount, asgi_app

from .common import stub
from .llm import Mistral
from .stt import Whisper
from .tts import XTTS

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

class Prompt(BaseModel):
    text: str

@stub.function()
@asgi_app()
def web():
    web_app = FastAPI()
    stt = Whisper()
    llm = Mistral()
    tts = XTTS()

    @web_app.get("/")
    def root():
        return {"message": "Hello, World"}

    @web_app.post("/voice_response")
    async def voice_respond(prompt: Prompt):
        audio = await prompt.text
        transcript = stt.transcribe.remote(audio)
        response = llm.generate.remote(transcript)
        verbalise = tts.speak.remote(response)
        return Response(content=verbalise, media_type="audio/wav")
    
    @web_app.post("/text_response")
    async def text_respond(request: Request):
        body = await request.json()
        prompt = body["text"]
        response = llm.generate.remote(prompt)
        return Response(content=response, media_type="text/plain")
    
    # web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app