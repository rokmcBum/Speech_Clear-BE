from fastapi import FastAPI
from app.controllers import VoiceController

app = FastAPI(title="Speech Analysis API")

# 라우터 등록
app.include_router(VoiceController.router)
