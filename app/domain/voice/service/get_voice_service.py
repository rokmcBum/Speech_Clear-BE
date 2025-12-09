# app/domain/voice/service/get_voice_service.py
import os
import tempfile
from typing import Any, Dict, List

import librosa
import numpy as np
from sqlalchemy.orm import Session, joinedload

from app.domain.voice.model.voice import (
    Voice,
    VoiceParagraphFeedback,
    VoiceSegment,
    VoiceSegmentVersion,
)
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import download_file


def get_voice(voice_id: int, db: Session, user) -> Dict[str, Any]:
    """
    단일 voice 조회 (part별로 그룹화)
    """
    # voice 소유권 검증
    voice = verify_voice_ownership(voice_id, user, db)
    
    # segments 조회 (order_no 순서대로, versions도 함께 로드)
    segments = (
        db.query(VoiceSegment)
        .options(joinedload(VoiceSegment.versions))
        .filter(VoiceSegment.voice_id == voice_id)
        .order_by(VoiceSegment.order_no.asc())
        .all()
    )
    
    # 문단별 피드백 조회
    paragraph_feedbacks = (
        db.query(VoiceParagraphFeedback)
        .filter(VoiceParagraphFeedback.voice_id == voice_id)
        .all()
    )
    
    # part별 피드백 매핑
    paragraph_feedback_map = {pf.part: pf.feedback for pf in paragraph_feedbacks}
    
    # category_name 설정
    if voice.category_id is None:
        category_name = "모든 speeches"
    else:
        category_name = voice.category.name if voice.category else "모든 speeches"
    
    # 원본 오디오 파일 다운로드 (dB_list 계산용)
    original_audio_path = None
    y_audio = None
    sr_audio = None
    
    try:
        # 임시 파일 생성
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(voice.filename)[1])
        original_audio_path = tmp_file.name
        tmp_file.close()
        
        # 오디오 다운로드
        download_file(voice.original_url, original_audio_path)
        
        # 오디오 로드
        y_audio, sr_audio = librosa.load(original_audio_path, sr=16000)
    except Exception as e:
        print(f"⚠️ 원본 오디오 다운로드/로드 실패: {e}")
        if original_audio_path and os.path.exists(original_audio_path):
            try:
                os.unlink(original_audio_path)
            except Exception:
                pass
    
    # segments를 order_no 순서대로 처리하고, 연속된 같은 part를 하나의 그룹으로 묶기
    # segments는 이미 order_no 오름차순으로 정렬되어 있음
    scripts = []
    current_part = None
    current_segments = []
    
    for seg in segments:
        # dB_list는 DB에 저장된 값 사용 (없으면 계산)
        dB_list = seg.db_list if seg.db_list else []
        
        # DB에 저장된 값이 없으면 계산 (하위 호환성)
        if not dB_list and y_audio is not None and sr_audio is not None:
            seg_start = int(seg.start_time * sr_audio)
            seg_end = int(seg.end_time * sr_audio)
            seg_audio = y_audio[seg_start:seg_end]
            
            if len(seg_audio) > 0:
                # 0.1초 간격으로 dB 계산
                interval_samples = int(0.1 * sr_audio)  # 0.1초에 해당하는 샘플 수
                for i in range(0, len(seg_audio), interval_samples):
                    chunk = seg_audio[i:i + interval_samples]
                    if len(chunk) > 0:
                        rms = librosa.feature.rms(y=chunk)
                        db_value = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))
                        dB_list.append(round(db_value, 2))
        
        part = seg.part if seg.part else "기타"
        
        # db, pitch 변환 (기존 로직 유지)
        # 참고: seg.db는 이미 저장된 값이므로 그대로 사용
        real_db = seg.db if seg.db else 0.0
        # pitch_mean_hz는 semitone이 아닌 Hz로 저장되어 있으므로 그대로 사용
        real_hz = seg.pitch_mean_hz if seg.pitch_mean_hz else 0.0

        # versions 정보 구성 (version_no 순서대로 정렬)
        versions_data = []
        if seg.versions:
            sorted_versions = sorted(seg.versions, key=lambda v: v.version_no)
            for version in sorted_versions:
                version_dB_list = version.db_list if version.db_list else []
                versions_data.append({
                    "id": version.id,
                    "version_no": version.version_no,
                    "text": version.text,
                    "segment_url": version.segment_url,
                    "feedback": version.feedback if version.feedback else "",
                    "dB_list": version_dB_list,
                    "created_at": version.created_at.isoformat() if version.created_at else None,
                    "metrics": {
                        "dB": version.db if version.db else 0.0,
                        "pitch_mean_hz": version.pitch_mean_hz if version.pitch_mean_hz else 0.0,
                        "rate_wpm": version.rate_wpm if version.rate_wpm else 0.0,
                        "pause_ratio": version.pause_ratio if version.pause_ratio else 0.0,
                        "prosody_score": version.prosody_score if version.prosody_score else 0.0,
                    }
                })

        segment_data = {
            "segment_id": seg.id,
            "text": seg.text,
            "start": seg.start_time,
            "end": seg.end_time,
            "segment_url": seg.segment_url,
            "feedback": seg.feedback if seg.feedback else "",
            "dB_list": dB_list,
            "versions": versions_data,  # versions 추가
            "metrics": {
                "dB": real_db if real_db else 0.0,
                "pitch_mean_hz": real_hz if real_hz else 0.0,
                "rate_wpm": seg.rate_wpm if seg.rate_wpm else 0.0,
            }
        }
        
        # part가 변경되면 이전 part 그룹을 저장하고 새 그룹 시작
        if current_part is not None and current_part != part:
            scripts.append({
                "part": current_part,
                "paragraph_feedback": paragraph_feedback_map.get(current_part, ""),
                "segments": current_segments
            })
            current_segments = []
        
        current_part = part
        current_segments.append(segment_data)
    
    # 마지막 part 그룹 추가
    if current_part is not None and current_segments:
        scripts.append({
            "part": current_part,
            "paragraph_feedback": paragraph_feedback_map.get(current_part, ""),
            "segments": current_segments
        })
    
    # 임시 파일 정리
    if original_audio_path and os.path.exists(original_audio_path):
        try:
            os.unlink(original_audio_path)
        except Exception:
            pass
    
    return {
        "voice_name": voice.name,
        "category_name": category_name,
        "voice_created_at": voice.created_at.isoformat() if voice.created_at else None,
        "voice_duration": voice.duration_sec if voice.duration_sec else 0.0,
        "scripts": scripts
    }

