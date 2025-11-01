import bcrypt
from fastapi import Form, Depends, HTTPException
from requests import Session

from app.domain.user.model.user import User
from app.infrastructure.db.db import get_session
from app.utils.encryption import decrypt_text
from app.utils.jwt_util import create_access_token


def login(email: str = Form(...),
          password: str = Form(...),
          db: Session = Depends(get_session)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="가입되지 않은 이메일입니다.")

    if not bcrypt.checkpw(password.encode("utf-8"), user.password.encode("utf-8")):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")

    token_data = {"user_id": user.id, "email": user.email}
    access_token = create_access_token(token_data)

    return {
        "message": "로그인 성공",
        "user": {
            "id": user.id,
            "name": decrypt_text(user.name),
            "email": user.email
        },
        "access_token": access_token,
        "token_type": "bearer"
    }
