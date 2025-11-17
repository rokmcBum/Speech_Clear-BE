# -*- coding: utf-8 -*-
"""
Voice 리소스에 대한 권한 검증 유틸리티
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice


def verify_voice_ownership(voice_id: int, user: User, db: Session) -> Voice:
    """
    voice 소유권을 검증하고 Voice 객체를 반환합니다.
    
    Args:
        voice_id: 검증할 voice ID
        user: 현재 사용자
        db: 데이터베이스 세션
    
    Returns:
        Voice: 검증된 Voice 객체
    
    Raises:
        HTTPException: voice가 없거나 소유자가 아닌 경우
    """
    voice = db.query(Voice).filter(Voice.id == voice_id).first()
    
    if not voice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice not found"
        )
    
    # 타입 확인 및 비교 (정수로 변환하여 비교)
    voice_user_id = int(voice.user_id) if voice.user_id is not None else None
    current_user_id = int(user.id) if user.id is not None else None
    
    if voice_user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to access this voice (voice.user_id={voice_user_id}, user.id={current_user_id})"
        )
    
    return voice

