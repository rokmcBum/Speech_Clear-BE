import asyncio
from typing import Dict, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.domain.category.model.category import Category
from app.domain.user.model.user import User
from app.domain.voice.service.compare_voice_feedback_service import (
    compare_voice_feedback,
)
from app.domain.voice.service.delete_voice_service import delete_voice
from app.domain.voice.service.get_my_voices_service import get_my_voices
from app.domain.voice.service.get_voice_service import get_voice
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
    if category_id == 0 or category_id is None:
        category_id = None
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
async def re_record(segment_id: int,
              file: UploadFile = File(...),
              db_list: Optional[str] = Form(None, description="0.1초 간격으로 측정된 dB 값 리스트 (JSON 문자열)"),
              db: Session = Depends(get_session),
              user: User = Depends(get_current_user)):
    """
    세그먼트 재녹음 (인증 필요)
    - 진행률 추적: GET /voice/progress/{user_id} (SSE)로 실시간 진행률 확인 가능
    """
    # 진행률 콜백 함수
    def update_progress(percentage: int):
        progress_store[user.id] = percentage

    # 초기화
    progress_store[user.id] = 0
    
    # 파일을 미리 읽어서 바이트로 변환
    file_content = await file.read()
    
    # 동기 함수를 별도 스레드에서 실행
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: re_record_segment(db, file, segment_id, user, db_list, update_progress, file_content)
        )
    finally:
        pass
    
    return result

@router.post("/synthesize/{voice_id}/")
def synthesize_speech(
    voice_id: int,
    body: dict = Body(...),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    음성 합성
    Body: {
        "selections": [
            {"segment_id": int, "selected_version_index": int},  # -1: 원본, 0: 첫 번째 재녹음, 1: 두 번째 재녹음, ...
        ]
    }
    """
    selections = body.get("selections", [])
    
    # 진행률 콜백 함수
    def update_progress(percentage: int):
        progress_store[user.id] = percentage

    # 초기화
    progress_store[user.id] = 0

    try:
        result = synthesize_voice(voice_id, db, user, selections, progress_callback=update_progress)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


@router.delete("/{voice_id}")
def delete_voice_endpoint(
    voice_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    음성 삭제 (인증 필요)
    - 소유권 검증 후 삭제
    - CASCADE로 segments, versions 등이 자동 삭제됨
    """
    result = delete_voice(voice_id, user, db)
    return result


@router.get("/{voice_id}/compare-feedback")
def compare_feedback_endpoint(
    voice_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    원본과 합성본의 total_feedback 비교
    - voice_id가 원본이면 최신 합성본과 비교
    - voice_id가 합성본이면 원본과 비교
    """
    result = compare_voice_feedback(voice_id, user, db)
    return result