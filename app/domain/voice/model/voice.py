from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, Boolean, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.infrastructure.db.db import Base


class Voice(Base):
    __tablename__ = "voices"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="CASCADE"), nullable=True)
    previous_voice_id = Column(Integer, nullable=True)
    name = Column(String(255), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    original_url = Column(String, nullable=False)
    duration_sec = Column(Float)
    sentence_feedback = Column(JSONB)  # 원본 voice의 sentence_feedback (재녹음 피드백 생성용)
    total_feedback = Column(String)  # 전체 음성에 대한 총괄 피드백
    created_at = Column(TIMESTAMP, server_default=func.now())
    segments = relationship("VoiceSegment", back_populates="voice")
    category = relationship("Category", back_populates="voices")


class VoiceSegment(Base):
    __tablename__ = "voice_segments"
    id = Column(Integer, primary_key=True)
    voice_id = Column(Integer, ForeignKey("voices.id"))
    order_no = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    part = Column(String(50), nullable=True)  # 문단 구분 (서론, 본론1, 본론2, 결론 등)
    start_time = Column(Float)
    end_time = Column(Float)
    segment_url = Column(String)
    db = Column(Float)
    pitch_mean_hz = Column(Float)
    rate_wpm = Column(Float)
    pause_ratio = Column(Float)
    prosody_score = Column(Float)
    feedback = Column(String)
    db_list = Column(JSONB)  # 0.1초 간격으로 측정된 dB 값 리스트
    last_re_analyzed_metrics = Column(JSONB)  # 최신 재녹음 분석 결과 (다음 재녹음 시 원본으로 사용)

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
    db_list = Column(JSONB)  # 0.1초 간격으로 측정된 dB 값 리스트
    created_at = Column(TIMESTAMP, server_default=func.now())

    segment = relationship("VoiceSegment", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("segment_id", "version_no", name="uq_segment_version"),
    )
