from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException
from requests import Session

from app.domain.user.model.user import User
from app.domain.voice.service.rerecord_voice_service import re_record_segment
from app.domain.voice.service.synthesize_voice_service import synthesize_voice
from app.domain.voice.service.upload_voice_service import process_voice
from app.infrastructure.db.db import get_session
from app.utils.jwt_util import get_current_user

router = APIRouter(
    prefix="/voice",
    tags=["voice"]
)


@router.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...), 
    category_id: int = Query(...),
    db: Session = Depends(get_session), 
    user: User = Depends(get_current_user)
):
    """
    음성 파일 업로드 후 분석 → JSON 결과 리턴
    - LLM을 사용하여 문단별(서론/본론/결론)로 분할
    """
    result = process_voice(db, file, user, category_id)
    return result


@router.post("/segment/{segment_id}/re_record")
def re_record(segment_id: int,
              file: UploadFile = File(...),
              db: Session = Depends(get_session),
              user: User = Depends(get_current_user)):
    result = re_record_segment(db, segment_id, file, user)
    return result

@router.post("/synthesize/{voice_id}/")
def synthesize_speech(voice_id: int,
              db: Session = Depends(get_session),
                      user: User = Depends(get_current_user)):
    result = synthesize_voice(voice_id, db,user)
    return result