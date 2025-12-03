# app/domain/voice/service/get_voice_service.py
import librosa
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import download_file
import tempfile
import os


def get_voice(voice_id: int, db: Session, user) -> Dict[str, Any]:
    """
    단일 voice 조회 (part별로 그룹화)
    """
    # voice 소유권 검증
    voice = verify_voice_ownership(voice_id, user, db)
    
    # segments 조회 (order_no 순서대로)
    segments = (
        db.query(VoiceSegment)
        .filter(VoiceSegment.voice_id == voice_id)
        .order_by(VoiceSegment.order_no.asc())
        .all()
    )
    
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
        # dB_list 계산 (0.1초마다)
        dB_list = []
        if y_audio is not None and sr_audio is not None:
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
                        db_value = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))
                        dB_list.append(round(db_value, 2))
        
        part = seg.part if seg.part else "기타"
        
        segment_data = {
            "segment_id": seg.id,
            "text": seg.text,
            "start": seg.start_time,
            "end": seg.end_time,
            "segment_url": seg.segment_url,
            "feedback": seg.feedback if seg.feedback else "",
            "dB_list": dB_list,
            "metrics": {
                "dB": seg.db if seg.db else 0.0,
                "pitch_mean_hz": seg.pitch_mean_hz if seg.pitch_mean_hz else 0.0,
                "rate_wpm": seg.rate_wpm if seg.rate_wpm else 0.0,
            }
        }
        
        # part가 변경되면 이전 part 그룹을 저장하고 새 그룹 시작
        if current_part is not None and current_part != part:
            scripts.append({
                "part": current_part,
                "segments": current_segments
            })
            current_segments = []
        
        current_part = part
        current_segments.append(segment_data)
    
    # 마지막 part 그룹 추가
    if current_part is not None and current_segments:
        scripts.append({
            "part": current_part,
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

