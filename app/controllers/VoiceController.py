from fastapi import UploadFile, File
from app.domain.voice.service.VoiceService import process_voice


async def analyze(file: UploadFile = File(...)):
    result = process_voice(file)
    return result
