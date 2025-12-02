import asyncio
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.domain.category.model.category import Category
from app.domain.user.model.user import User
from app.domain.voice.service.get_my_voices_service import get_my_voices
from app.domain.voice.service.rerecord_voice_service import re_record_segment
from app.domain.voice.service.synthesize_voice_service import synthesize_voice
from app.domain.voice.service.upload_voice_service import process_voice
from app.infrastructure.db.db import get_session
from app.utils.jwt_util import get_current_user

router = APIRouter(
    prefix="/voice",
    tags=["voice"]
)

# 진행률 저장소 (user_id -> progress percentage)
progress_store: Dict[int, int] = {}

@router.get("/progress/{user_id}")
async def get_progress(user_id: int):
    async def event_generator():
        while True:
            progress = progress_store.get(user_id, 0)
            yield f"data: {progress}\n\n"
            if progress >= 100:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/list")
def list_voices(
    category_id: int = Query(None, description="카테고리 ID (선택적, 없으면 전체 조회)"),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    내 음성 목록 조회 (인증 필요)
    - category_id가 제공되면 해당 카테고리의 음성만 조회
    - 없으면 전체 음성 조회
    """
    voices = get_my_voices(user, db, category_id)
    return {"voices": voices}


@router.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...), 
    category_id: Optional[int] = Form(default=None),
    name: str = Form(...),
    db: Session = Depends(get_session), 
    user: User = Depends(get_current_user)
):
    """
    음성 파일 업로드 후 분석 → JSON 결과 리턴
    - LLM을 사용하여 문단별(서론/본론/결론)로 분할
    - name: 음성 이름
    - category_id: 카테고리 ID (선택적, 0이면 NULL로 저장)
    """
    # category_id가 0이면 기본 카테고리(첫 번째) 할당
    if category_id == 0:
        first_category = db.query(Category).filter(Category.user_id == user.id).first()
        if not first_category:
            raise HTTPException(status_code=400, detail="생성된 카테고리가 없습니다.")
        category_id = first_category.id

    # 진행률 콜백 함수
    def update_progress(percentage: int):
        progress_store[user.id] = percentage

    # 초기화
    progress_store[user.id] = 0
    
    try:
        result = process_voice(db, file, user, category_id, name, progress_callback=update_progress)
    finally:
        # 완료 후 정리 (선택적, 클라이언트가 100%를 받을 시간 여유를 위해 바로 삭제하지 않거나, 
        # SSE 연결이 끊길 때까지 유지할 수도 있음. 여기서는 100% 도달 후 SSE가 끊기도록 둠)
        pass
        
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
    result = synthesize_voice(voice_id, db, user)
    return result