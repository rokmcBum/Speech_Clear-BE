# -*- coding: utf-8 -*-
"""
음성 삭제 서비스
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice
from app.domain.voice.utils.voice_permission import verify_voice_ownership


def delete_voice(voice_id: int, user: User, db: Session):
    """
    음성 삭제
    - 소유권 검증 후 삭제
    - CASCADE로 segments, versions 등이 자동 삭제됨
    """
    # voice 소유권 검증
    voice = verify_voice_ownership(voice_id, user, db)
    
    # Voice 삭제 (CASCADE로 관련 데이터 자동 삭제)
    db.delete(voice)
    db.commit()
    
    return {
        "message": "음성이 성공적으로 삭제되었습니다.",
        "voice_id": voice_id
    }

