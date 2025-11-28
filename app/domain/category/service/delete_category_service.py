# -*- coding: utf-8 -*-
"""
카테고리 삭제 서비스
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.category.model.category import Category
from app.domain.user.model.user import User


def delete_category(category_id: int, user: User, db: Session):
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
            detail="이 카테고리를 삭제할 권한이 없습니다."
        )
    
    db.delete(category)
    db.commit()
    
    return {
        "message": "카테고리가 성공적으로 삭제되었습니다.",
        "id": category_id
    }

