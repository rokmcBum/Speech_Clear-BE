from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, Boolean, func, UniqueConstraint
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
    segment_url = Column(String)
    db = Column(Float)
    pitch_mean_hz = Column(Float)
    rate_wpm = Column(Float)
    pause_ratio = Column(Float)
    prosody_score = Column(Float)
    feedback = Column(String)

    voice = relationship("Voice", back_populates="segments")
    versions = relationship("VoiceSegmentVersion", back_populates="segment", cascade="all, delete-orphan")


class VoiceSegmentVersion(Base):
    __tablename__ = "voice_segment_versions"
    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("voice_segments.id"), nullable=False)
    version_no = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    segment_url = Column(String, nullable=False)
    db = Column(Float)
    pitch_mean_hz = Column(Float)
    rate_wpm = Column(Float)
    pause_ratio = Column(Float)
    prosody_score = Column(Float)
    feedback = Column(String)
    created_at = Column(TIMESTAMP, server_default=func.now())
    is_selected = Column(Boolean, default=False)

    segment = relationship("VoiceSegment", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("segment_id", "version_no", name="uq_segment_version"),
    )
