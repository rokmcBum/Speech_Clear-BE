import re
from fastapi import Form, Depends, HTTPException, APIRouter
from pydantic import EmailStr
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.service.register_user_service import register_user
from app.infrastructure.db.db import get_session

router = APIRouter()

EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"


@router.post("/register")
async def register_users(
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_session)
):
    # 1️⃣ 입력값 누락 체크
    if not name or not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="모든 필드를 입력해주세요."
        )

    # 2️⃣ 이메일 형식 검증
    if not re.match(EMAIL_REGEX, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이메일 형식이 올바르지 않습니다."
        )

    # 3️⃣ 회원가입 로직 실행
    result = register_user(name, email, password, db)
    return result
