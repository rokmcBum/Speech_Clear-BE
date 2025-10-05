# app/domain/voice/models.py
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, func
from sqlalchemy.orm import relationship

from app.infrastructure.db.db import Base


class Voice(Base):
    __tablename__ = "voices"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    original_url = Column(String, nullable=False)
    duration_sec = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())
    segments = relationship("VoiceSegment", back_populates="voice")


class VoiceSegment(Base):
    __tablename__ = "voice_segments"
    id = Column(Integer, primary_key=True)
    voice_id = Column(Integer, ForeignKey("voices.id"))
    order_no = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    start_time = Column(Float)
    end_time = Column(Float)
    segment_url = Column(String)   # 👈 segment 파일 Object Storage URL
    db = Column(Float)
    pitch_mean_hz = Column(Float)
    rate_wpm = Column(Float)
    pause_ratio = Column(Float)
    prosody_score = Column(Float)
    feedback = Column(String)
    voice = relationship("Voice", back_populates="segments")
