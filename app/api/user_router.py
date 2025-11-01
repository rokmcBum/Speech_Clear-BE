import re

from fastapi import Form, Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.user.service.login_user_service import login
from app.domain.user.service.register_user_service import register_user
from app.infrastructure.db.db import get_session
from app.utils.jwt_util import get_current_user

router = APIRouter()

EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"


@router.post("/register")
async def register_users(
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_session)
):
    if not name or not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="모든 필드를 입력해주세요."
        )

    if not re.match(EMAIL_REGEX, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이메일 형식이 올바르지 않습니다."
        )

    result = register_user(name, email, password, db)
    return result


@router.post("/login")
async def login_user(
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_session)
):
    # 1️⃣ 입력 검증
    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이메일과 비밀번호를 모두 입력해주세요."
        )

    return login(email, password, db)


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email
    }
