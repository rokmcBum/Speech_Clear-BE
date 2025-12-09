import os
import tempfile
import json
from typing import Optional, List

import librosa
import numpy as np
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.user.model.user import User
from app.domain.voice.model.voice import VoiceSegment, VoiceSegmentVersion
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import upload_file
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
    make_part_index_map
)


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _next_version_no(db: Session, segment_id: int) -> int:
    last = (
        db.query(VoiceSegmentVersion)
        .filter(VoiceSegmentVersion.segment_id == segment_id)
        .order_by(VoiceSegmentVersion.version_no.desc())
        .first()
    )
    return (last.version_no + 1) if last else 1


def re_record_segment(db: Session, segment_id: int, file: UploadFile, user: User, db_list_str: Optional[str] = None):
    seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
    
    if not seg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    
    # voice 소유권 검증
    verify_voice_ownership(seg.voice_id, user, db)

    ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    ver_no = _next_version_no(db, segment_id)
    object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
    seg_url = upload_file(tmp_path, object_name)

    # Clova Speech API로 STT 수행
    clova_result = make_voice_to_stt(tmp_path)
    full_text = clova_result["text"]
    clova_words = clova_result.get("words", [])  # Clova Speech의 word timestamps
    
    # librosa로 오디오 분석
    y, sr = librosa.load(tmp_path, sr=16000)
    frame_length = 2048
    hop_length = 256

    # RMS (음성 크기)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]  # (1,T) -> (T,)

    # F0 (음성 높낮이)
    f0, _, _ = librosa.pyin(
        y,
        fmin=80,
        fmax=300,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=sr
    )

    # 프레임 시간 (초)
    frame_idx = np.arange(rms.shape[0])
    frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr

    # 전체 오디오 길이 (초)
    duration = float(len(y) / sr)
    
    # Clova segments에서 첫 번째 segment 사용 (한 문장만 처리)
    clova_segments = clova_result.get("segments", [])
    if not clova_segments:
        # segments가 없으면 전체를 하나의 segment로 처리
        seg_start_ms = 0
        seg_end_ms = int(duration * 1000)
        seg_text = full_text
        seg_words = clova_words
    else:
        clova_seg = clova_segments[0]
        seg_start_ms = clova_seg.get("start", 0)
        seg_end_ms = clova_seg.get("end", int(duration * 1000))
        seg_text = clova_seg.get("text", full_text).strip()
        seg_words = clova_seg.get("words", clova_words)
    
    # 초 단위로 변환
    seg_start = seg_start_ms / 1000.0
    seg_end = seg_end_ms / 1000.0

    # 이 문장에 속하는 프레임 인덱스
    y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)

    # STT 결과 기반 유성 마스크 생성
    final_segments_for_mask = [{
        "words": [[w[0], w[1], w[2]] for w in seg_words]  # [start_ms, end_ms, text] 형태
    }]
    full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, final_segments_for_mask)

    # 이 문장 구간 + 유성 마스크 둘 다 만족하는 프레임
    seg_voice_masked = y_seg & full_voice_masked

    rms_seg = rms[seg_voice_masked]
    f0_seg = f0[seg_voice_masked]

    # dB 계산
    mean_r, std_r, cv_energy = compute_energy_stats_segment(
        rms=rms_seg,
        silence_thresh=1e-6
    )

    # pitch 계산
    mean_st, std_st, cv_pitch = compute_pitch_cv_segment(
        f0_hz=f0_seg,
        f0_min=1e-3
    )
    
    # segment 프레임 시간 (초)
    seg_frame_idx = np.arange(rms[y_seg].shape[0])
    seg_frame_times = (seg_frame_idx * hop_length + hop_length / 2.0) / sr

    # segment에 대한 문장 끝 경계 특징 계산
    final_rms_ratio, final_rms_slope, final_pitch_semitone_drop, final_pitch_semitone_slope = compute_final_boundary_features_for_segment(
        rms=rms[y_seg],
        f0_hz=f0[y_seg],
        voice_masked=full_voice_masked[y_seg],
        frame_times=seg_frame_times,
        seg_length=seg_end-seg_start
    )

    # 말하기 속도 계산 (wpm)
    words_count = len(seg_text.split())
    duration_min = (seg_end - seg_start) / 60
    rate_wpm = words_count / duration_min if duration_min > 0 else 0

    # segment_info 구성 (upload_voice_service.py와 동일한 형식)
    segment_info = {
        "id": 0,
        "part": seg.part,  # 기존 segment의 part 사용
        "text": seg_text,
        "start": seg_start,
        "end": seg_end,
        "energy": {
            "mean_rms": round(mean_r, 2) if not np.isnan(mean_r) else 0.0,
            "std_rms": round(std_r, 2) if not np.isnan(std_r) else 0.0,
            "cv": round(cv_energy, 4) if not np.isnan(cv_energy) else 0.0
        },
        "pitch": {
            "mean_st": round(mean_st, 2) if not np.isnan(mean_st) else 0.0,
            "std_st": round(std_st, 2) if not np.isnan(std_st) else 0.0,
            "cv": round(cv_pitch, 4) if not np.isnan(cv_pitch) else 0.0
        },
        "wpm": {
            "word_count": words_count,
            "rate_wpm": round(rate_wpm, 1),
            "duration_sec": round(seg_end - seg_start, 3)
        },
        "final_boundary": {
            "final_rms_ratio": round(final_rms_ratio, 2) if not np.isnan(final_rms_ratio) else 0.0,
            "final_rms_slope": round(final_rms_slope, 4) if not np.isnan(final_rms_slope) else 0.0,
            "final_pitch_semitone_drop": round(final_pitch_semitone_drop, 2) if not np.isnan(final_pitch_semitone_drop) else 0.0,
            "final_pitch_semitone_slope": round(final_pitch_semitone_slope, 4) if not np.isnan(final_pitch_semitone_slope) else 0.0
        },
        "words": []
    }

    # 단어 단위 분석
    for w in seg_words:
        w_text = w[2].strip()
        w_start, w_end = w[0]/1000, w[1]/1000
        w_start_samp, w_end_samp = int(w_start*sr), int(w_end*sr)
        y_word = y[w_start_samp:w_end_samp]

        if len(y_word) == 0:
            continue

        # --- dB
        w_rms = librosa.feature.rms(y=y_word)
        db_value = float(np.mean(librosa.amplitude_to_db(w_rms, ref=1.0)))

        # --- pitch
        w_f0, _, _ = librosa.pyin(
            y_word,
            fmin=80,
            fmax=300,
            sr=sr
        )
        pitch_vals = w_f0[~np.isnan(w_f0)]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0

        duration_word = w_end - w_start

        segment_info["words"].append({
            "text": w_text,
            "start": w_start,
            "end": w_end,
            "metrics": {
                "dB": round(db_value, 2),
                "pitch_mean_hz": round(pitch_mean, 2),
                "pitch_std_hz": round(pitch_std, 2),
                "duration_sec": round(duration_word, 3)
            }
        })

    # 문단별 인덱스 맵 생성 (한 문장만 있으므로 간단하게)
    paragraph_index = make_part_index_map([segment_info])

    # 피드백 생성
    feedbacks_list, paragraph_feedback = make_feedback_service.make_feedback([segment_info], paragraph_index)
    
    # 피드백 추출 (한 문장만 있으므로 첫 번째 피드백 사용)
    feedback_text = ""
    if feedbacks_list and len(feedbacks_list) > 0:
        feedback_text = feedbacks_list[0].get("feedback", "")

    # dB_list는 frontend에서 받은 값을 그대로 사용
    dB_list = []
    if db_list_str:
        try:
            dB_list = json.loads(db_list_str)
            # float 리스트로 변환
            dB_list = [float(x) for x in dB_list] if isinstance(dB_list, list) else []
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"⚠️ dB_list 파싱 실패: {e}, 빈 리스트로 저장")
            dB_list = []

    # DB에 저장
    energy = segment_info.get("energy", {})
    pitch = segment_info.get("pitch", {})
    wpm = segment_info.get("wpm", {})

    version = VoiceSegmentVersion(
        segment_id=seg.id,
        version_no=ver_no,
        text=seg_text,
        segment_url=seg_url,
        db=float(energy.get("mean_rms", 0.0)),  # mean_rms를 dB로 사용
        pitch_mean_hz=float(pitch.get("mean_st", 0.0)),  # semitone을 Hz로 저장 (실제로는 semitone이지만 호환성 유지)
        rate_wpm=float(wpm.get("rate_wpm", 0.0)),
        pause_ratio=0.0,  # 새로운 구조에는 없음
        prosody_score=0.0,  # 새로운 구조에는 없음
        feedback=feedback_text,
        db_list=dB_list,  # 0.1초 간격으로 측정된 dB 값 리스트
    )
    db.add(version)

    db.commit()
    db.refresh(version)

    return {
        "id": version.id,
        "segment_id": version.segment_id,
        "version_no": version.version_no,
        "text": version.text,
        "segment_url": version.segment_url,
        "feedback": version.feedback,
        "dB_list": dB_list,  # 0.1초 간격으로 측정된 dB 값 리스트
        "metrics": {
            "dB": version.db,
            "pitch_mean_hz": version.pitch_mean_hz,
            "rate_wpm": version.rate_wpm,
            "pause_ratio": version.pause_ratio,
            "prosody_score": version.prosody_score,
        }
    }
