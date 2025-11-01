from sqlalchemy import Column, Integer, String, TIMESTAMP, func
from app.infrastructure.db.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
