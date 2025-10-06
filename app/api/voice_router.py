from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException
from requests import Session

from app.domain.voice.service.rerecord_voice_service import re_record_segment
from app.domain.voice.service.upload_voice_service import process_voice
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


@router.post("/segment/{segment_id}/re_record")
def re_record(segment_id: int,
              file: UploadFile = File(...),
              db: Session = Depends(get_session)):
    result = re_record_segment(db, segment_id, file)
    if not result:
        raise HTTPException(status_code=404, detail="Segment not found")
    return result