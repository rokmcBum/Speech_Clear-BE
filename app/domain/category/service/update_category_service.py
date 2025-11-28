# -*- coding: utf-8 -*-
"""
카테고리 수정 서비스
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.category.model.category import Category
from app.domain.user.model.user import User


def update_category(category_id: int, name: str, user: User, db: Session):
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="카테고리 이름을 입력해주세요."
        )
    
    category = db.query(Category).filter(Category.id == category_id).first()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="카테고리를 찾을 수 없습니다."
        )
    
    # 소유권 검증
    if category.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="이 카테고리를 수정할 권한이 없습니다."
        )
    
    # 이름 업데이트
    category.name = name.strip()
    db.commit()
    db.refresh(category)
    
    return {
        "id": category.id,
        "user_id": category.user_id,
        "name": category.name,
        "created_at": category.created_at.isoformat() if category.created_at else None
    }

