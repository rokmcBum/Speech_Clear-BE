from fastapi import Form, Depends, HTTPException
from requests import Session
from starlette import status

from app.domain.user.model.user import User
from app.infrastructure.db.db import get_session
import bcrypt

from app.utils.encryption import encrypt_text, decrypt_text


def register_user(name: str = Form(...),
                  email: str = Form(...),
                  password: str = Form(...), db: Session = Depends(get_session)):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미 가입된 이메일입니다.")

    encrypted_name = encrypt_text(name)
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # 4️⃣ DB 저장
    new_user = User(name=encrypted_name, email=email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "회원가입이 완료되었습니다.",
        "user": {
            "id": new_user.id,
            "name": decrypt_text(new_user.name),
            "email": new_user.email
        }
    }
