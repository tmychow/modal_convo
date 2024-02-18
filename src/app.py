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

@asgi_app()
def web():
    web_app = FastAPI()
    stt = Whisper()
    llm = Mistral()
    tts = XTTS()

    @web_app.post("/voice_response")
    def voice_respond(request: Request):
        audio = request.body()
        transcript = stt.transcribe(audio)
        response = llm.generate(transcript)
        verbalise = tts.speak(response)
        return Response(content=verbalise, media_type="audio/wav")
    
    @web_app.post("/text_response")
    def text_respond(request: Request):
        text = request.body()
        response = llm.generate(text)
        return Response(content=response, media_type="text/plain")
    
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app