# -*- coding: utf-8 -*-
"""
회원 탈퇴 서비스
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User


def delete_user(user: User, db: Session):
    """
    회원 탈퇴
    - CASCADE로 categories, voices 등이 자동 삭제됨
    """
    # 사용자 삭제 (CASCADE로 관련 데이터 자동 삭제)
    db.delete(user)
    db.commit()
    
    return {
        "message": "회원 탈퇴가 완료되었습니다.",
        "user_id": user.id
    }

