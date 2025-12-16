# -*- coding: utf-8 -*-
"""
원본과 합성본의 total_feedback 비교 서비스
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice
from app.domain.voice.utils.voice_permission import verify_voice_ownership


def compare_voice_feedback(voice_id: int, user: User, db: Session):
    """
    원본과 합성본의 total_feedback 비교
    - voice_id: 원본 또는 합성본 voice ID
    - 원본이면 합성본을 찾아서 비교
    - 합성본이면 원본을 찾아서 비교
    """
    # voice 소유권 검증
    voice = verify_voice_ownership(voice_id, user, db)
    
    original_voice = None
    synthesized_voice = None
    
    if voice.previous_voice_id:
        # 합성본인 경우
        synthesized_voice = voice
        original_voice = db.query(Voice).filter(Voice.id == voice.previous_voice_id).first()
        if not original_voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="원본 음성을 찾을 수 없습니다."
            )
        # 원본 소유권 검증
        if original_voice.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="원본 음성에 대한 권한이 없습니다."
            )
    else:
        # 원본인 경우, 합성본 찾기
        original_voice = voice
        synthesized_voice = db.query(Voice).filter(Voice.previous_voice_id == voice_id).order_by(Voice.created_at.desc()).first()
    
    return {
        "original": {
            "voice_id": original_voice.id,
            "name": original_voice.name,
            "original_url": original_voice.original_url if original_voice.original_url else "",
            "total_feedback": original_voice.total_feedback if original_voice.total_feedback else ""
        },
        "synthesized": {
            "voice_id": synthesized_voice.id if synthesized_voice else None,
            "name": synthesized_voice.name if synthesized_voice else None,
            "original_url": synthesized_voice.original_url if synthesized_voice and synthesized_voice.original_url else "",
            "total_feedback": synthesized_voice.total_feedback if synthesized_voice and synthesized_voice.total_feedback else ""
        } if synthesized_voice else None
    }

