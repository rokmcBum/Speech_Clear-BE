from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.category.service.create_category_service import create_category
from app.domain.category.service.get_my_categories_service import get_my_categories
from app.infrastructure.db.db import get_session
from app.utils.jwt_util import get_current_user

router = APIRouter(
    prefix="/category",
    tags=["category"]
)


@router.post("/")
async def create_category_endpoint(
    name: str = Form(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    카테고리 생성
    """
    result = create_category(name, current_user, db)
    return result


@router.get("/")
async def get_my_categories_endpoint(
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    현재 사용자의 모든 카테고리 조회
    """
    result = get_my_categories(current_user, db)
    return result

