from sqlalchemy.orm import Session
from sqlalchemy import func

from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.user.model.user import User


def get_my_voices(user: User, db: Session, category_id: int = None):
    """
    사용자의 음성 목록을 조회합니다.
    
    Args:
        user: 현재 사용자
        db: 데이터베이스 세션
        category_id: 카테고리 ID (선택적, None이면 전체 조회)
    
    Returns:
        List[Dict]: 음성 목록
    """
    # 기본 쿼리: 사용자의 음성만 조회
    query = db.query(Voice).filter(Voice.user_id == user.id)
    
    # 카테고리 필터링 (선택적)
    if category_id is not None:
        query = query.filter(Voice.category_id == category_id)
    
    # 생성일 기준 내림차순 정렬
    voices = query.order_by(Voice.created_at.desc()).all()
    
    # 각 voice의 세그먼트 개수 계산 및 미리보기 텍스트 추출
    result = []
    for voice in voices:
        segments_count = db.query(func.count(VoiceSegment.id)).filter(
            VoiceSegment.voice_id == voice.id
        ).scalar()
        
        # 모든 segments의 텍스트를 합쳐서 100자로 제한하여 미리보기 생성
        segments = db.query(VoiceSegment).filter(
            VoiceSegment.voice_id == voice.id
        ).order_by(VoiceSegment.order_no.asc()).all()
        
        preview_text = None
        if segments:
            # 모든 segments의 텍스트를 합치기
            full_text = " ".join([seg.text for seg in segments if seg.text])
            if full_text:
                preview_text = full_text[:100]
                if len(full_text) > 100:
                    preview_text += "..."
        
        result.append({
            "id": voice.id,
            "name": voice.name,
            "category_id": voice.category_id,
            "category_name": voice.category.name if voice.category else "모든 speech",
            "original_url": voice.original_url,
            "duration_sec": voice.duration_sec,
            "segments_count": segments_count,
            "preview_text": preview_text,
            "created_at": voice.created_at.isoformat() if voice.created_at else None
        })
    
    return result

