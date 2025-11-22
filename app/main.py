from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import voice_router, user_router, category_router

app = FastAPI(title="Speech Clear")

origins = [
    "http://localhost:5173",      # React 개발 서버
    "http://127.0.0.1:5173",      # 일부 브라우저에서 필요
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 정확한 origin만 허용!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(voice_router.router)
app.include_router(user_router.router)
app.include_router(category_router.router)