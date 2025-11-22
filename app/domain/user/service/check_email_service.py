from sqlalchemy.orm import Session

from app.domain.user.model.user import User


def check_email_availability(email: str, db: Session) -> dict:
    """
    이메일 중복 검증
    """
    existing = db.query(User).filter(User.email == email).first()
    
    if existing:
        return {
            "available": False,
            "message": "이미 사용 중인 이메일입니다."
        }
    else:
        return {
            "available": True,
            "message": "사용 가능한 이메일입니다."
        }

