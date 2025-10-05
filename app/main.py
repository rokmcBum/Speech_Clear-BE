from fastapi import FastAPI

from app.api import voice_router

app = FastAPI(title="Speech Clear")

# 라우터 등록
app.include_router(voice_router.router)
