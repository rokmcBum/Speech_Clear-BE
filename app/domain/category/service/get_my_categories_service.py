from sqlalchemy.orm import Session

from app.domain.category.model.category import Category
from app.domain.user.model.user import User


def get_my_categories(user: User, db: Session):
    categories = db.query(Category).filter(Category.user_id == user.id).order_by(Category.created_at.desc()).all()

    return [
        {
            "id": category.id,
            "user_id": category.user_id,
            "name": category.name,
            "created_at": category.created_at.isoformat() if category.created_at else None
        }
        for category in categories
    ]

