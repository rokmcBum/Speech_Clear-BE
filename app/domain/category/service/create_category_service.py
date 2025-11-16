from fastapi import Form, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.category.model.category import Category
from app.domain.user.model.user import User


def create_category(name: str, user: User, db: Session):
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="카테고리 이름을 입력해주세요."
        )

    new_category = Category(
        user_id=user.id,
        name=name.strip()
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)

    return {
        "id": new_category.id,
        "user_id": new_category.user_id,
        "name": new_category.name,
        "created_at": new_category.created_at.isoformat() if new_category.created_at else None
    }

