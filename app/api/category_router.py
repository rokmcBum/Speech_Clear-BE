from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.category.service.create_category_service import create_category
from app.domain.category.service.get_my_categories_service import get_my_categories
from app.domain.category.service.update_category_service import update_category
from app.domain.category.service.delete_category_service import delete_category
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


@router.put("/{category_id}")
async def update_category_endpoint(
    category_id: int,
    name: str = Form(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    카테고리 수정
    - category_id: 수정할 카테고리 ID
    - name: 새로운 카테고리 이름
    """
    result = update_category(category_id, name, current_user, db)
    return result


@router.delete("/{category_id}")
async def delete_category_endpoint(
    category_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    카테고리 삭제
    - category_id: 삭제할 카테고리 ID
    - 주의: 카테고리 삭제 시 연결된 모든 음성(voices)도 함께 삭제됩니다.
    """
    result = delete_category(category_id, current_user, db)
    return result

