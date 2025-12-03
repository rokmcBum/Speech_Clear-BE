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
from app.domain.voice.service.get_my_voices_service import get_my_voices
from app.domain.voice.service.get_voice_service import get_voice
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
        # 연결 시 기존에 100%인 상태라면 초기화 (새로운 요청으로 간주)
        if progress_store.get(user_id) == 100:
            progress_store[user_id] = 0
            
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
    
    # 파일을 미리 읽어서 바이트로 변환 (스레드에서 UploadFile 객체 접근 시 문제 발생 가능)
    file_content = await file.read()
    
    # 동기 함수인 process_voice를 별도 스레드에서 실행하여 이벤트 루프 차단 방지
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, 
            lambda: process_voice(db, file, user, category_id, name, progress_callback=update_progress, file_content=file_content)
        )
    finally:
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


@router.get("/{voice_id}")
def get_voice_detail(
    voice_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    단일 voice 상세 조회
    - part별로 그룹화된 segments 반환
    - 각 segment에 0.1초마다의 dB_list 포함
    - category_id가 NULL이면 category_name은 "모든 speeches"
    """
    result = get_voice(voice_id, db, user)
    return result