from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException, Form
from pydantic import EmailStr
from requests import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.user.service.register_user_service import register_user
from app.infrastructure.db.db import get_session

router = APIRouter(
    prefix="/user",
    tags=["user"]
)


@router.post("/register")
async def register_users(name: str = Form(...),
                         email: EmailStr = Form(...),
                         password: str = Form(...), db: Session = Depends(get_session)):
    if not name or not email or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="모든 필드를 입력해주세요.")

    result = register_user(name, email, password, db)
    return result
