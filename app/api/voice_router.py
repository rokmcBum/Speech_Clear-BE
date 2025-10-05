from fastapi import APIRouter, UploadFile, File, Depends
from requests import Session

from app.domain.voice_service import process_voice
from app.infrastructure.db.db import get_session

router = APIRouter(
    prefix="/voice",
    tags=["voice"]
)


@router.post("/analyze")
async def analyze_voice(file: UploadFile = File(...), db: Session = Depends(get_session)):
    """
    음성 파일 업로드 후 분석 → JSON 결과 리턴
    """
    result = process_voice(db ,file)
    return result
