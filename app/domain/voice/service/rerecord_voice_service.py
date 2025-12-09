import json
import os
import tempfile
from typing import Callable, List, Optional

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile
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
    make_part_index_map,
)


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def safe_float(value):
    """NaN/inf를 안전하게 처리하는 float 변환 함수"""
    if value is None:
        return 0.0
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    except (TypeError, ValueError):
        return 0.0


def _next_version_no(db: Session, segment_id: int) -> int:
    last = (
        db.query(VoiceSegmentVersion)
        .filter(VoiceSegmentVersion.segment_id == segment_id)
        .order_by(VoiceSegmentVersion.version_no.desc())
        .first()
    )
    return (last.version_no + 1) if last else 1


def re_record_segment(db: Session, file: UploadFile, segment_id: int, user: User, db_list_str: Optional[str] = None, progress_callback: Optional[Callable[[int], None]] = None, file_content: bytes = None):
    if progress_callback:
        progress_callback(10)  # 시작: 10%
    
    seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
    
    if not seg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    
    # voice 소유권 검증
    verify_voice_ownership(seg.voice_id, user, db)

    ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
        if file_content:
            tmp.write(file_content)
        else:
            tmp.write(file.file.read())
        tmp_path = tmp.name

    if progress_callback:
        progress_callback(20)  # 파일 저장 완료: 20%

    ver_no = _next_version_no(db, segment_id)
    object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
    seg_url = upload_file(tmp_path, object_name)

    if progress_callback:
        progress_callback(30)  # Object Storage 업로드 완료: 30%

    # Clova Speech API로 STT 수행
    clova_result = make_voice_to_stt(tmp_path)
    
    if progress_callback:
        progress_callback(40)  # Clova Speech 완료: 40%
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
    
    # [FIX] dB calculation: RMS -> dB (ref=1.0)
    mean_db = 0.0
    if len(rms_seg) > 0:
        mean_db = float(np.mean(librosa.amplitude_to_db(rms_seg, ref=1.0)))

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
        "start": safe_float(seg_start),
        "end": safe_float(seg_end),
        "energy": {
            "mean_rms": safe_float(mean_r),
            "std_rms": safe_float(std_r),
            "cv": safe_float(cv_energy),
            "mean_db": safe_float(mean_db)  # [FIX] Added mean_db
        },
        "pitch": {
            "mean_st": safe_float(mean_st),
            "std_st": safe_float(std_st),
            "cv": safe_float(cv_pitch),
            "mean_hz": safe_float(np.mean(f0_seg)) if len(f0_seg) > 0 else 0.0  # [FIX] Added mean_hz
        },
        "wpm": {
            "word_count": words_count,
            "rate_wpm": safe_float(rate_wpm),
            "duration_sec": safe_float(seg_end - seg_start)
        },
        "final_boundary": {
            "final_rms_ratio": safe_float(final_rms_ratio),
            "final_rms_slope": safe_float(final_rms_slope),
            "final_pitch_semitone_drop": safe_float(final_pitch_semitone_drop),
            "final_pitch_semitone_slope": safe_float(final_pitch_semitone_slope)
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
            "start": safe_float(w_start),
            "end": safe_float(w_end),
            "metrics": {
                "dB": safe_float(db_value),
                "pitch_mean_hz": safe_float(pitch_mean),
                "pitch_std_hz": safe_float(pitch_std),
                "duration_sec": safe_float(duration_word)
            }
        })

    # 문단별 인덱스 맵 생성 (한 문장만 있으므로 간단하게)
    paragraph_index = make_part_index_map([segment_info])

    if progress_callback:
        progress_callback(70)  # 오디오 분석 완료: 70%

    # 피드백 생성
    feedbacks_list, paragraph_feedback = make_feedback_service.make_feedback([segment_info], paragraph_index)
    
    if progress_callback:
        progress_callback(80)  # 피드백 생성 완료: 80%
    
    # 피드백 추출 (한 문장만 있으므로 첫 번째 피드백 사용)
    feedback_text = ""
    if feedbacks_list and len(feedbacks_list) > 0:
        feedback_text = feedbacks_list[0].get("feedback", "")

    # dB_list는 frontend에서 받은 값을 그대로 사용
    dB_list = []
    if db_list_str:
        try:
            dB_list = json.loads(db_list_str)
            # float 리스트로 변환 (NaN/inf 체크 포함)
            dB_list = [safe_float(x) for x in dB_list] if isinstance(dB_list, list) else []
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"⚠️ dB_list 파싱 실패: {e}, 빈 리스트로 저장")
            dB_list = []

    # DB에 저장
    energy = segment_info.get("energy", {})
    pitch = segment_info.get("pitch", {})
    wpm = segment_info.get("wpm", {})
    
    # real_db 계산: 세그먼트 오디오를 다시 로드하여 RMS를 dB로 변환 (upload_voice_service.py와 동일)
    real_db = 0.0
    real_hz = 0.0
    
    try:
        y_seg_audio, sr_seg = librosa.load(tmp_path, sr=16000)
        if len(y_seg_audio) > 0:
            # real_db 계산: 전체 세그먼트의 RMS를 dB로 변환 (upload_voice_service.py와 동일)
            rms_full = librosa.feature.rms(y=y_seg_audio)
            real_db = float(np.mean(librosa.amplitude_to_db(rms_full, ref=np.max)))
    except Exception as e:
        print(f"⚠️ 세그먼트 오디오 재로드 실패: {e}")
        # fallback: mean_db 사용
        real_db = safe_float(energy.get("mean_db", 0.0))
    
    # real_hz 계산: mean_st(semitone)을 Hz로 변환 (upload_voice_service.py와 동일)
    mean_st = pitch.get("mean_st", 0.0)
    if mean_st and not np.isnan(mean_st) and mean_st != 0.0:
        real_hz = 55.0 * (2 ** (mean_st / 12.0))
    else:
        # fallback: mean_hz가 이미 Hz 값이므로 그대로 사용
        real_hz = safe_float(pitch.get("mean_hz", 0.0))
    
    # NaN/inf 체크 및 float 변환 (JSON 직렬화를 위해)
    real_db = safe_float(real_db)
    real_hz = safe_float(real_hz)

    version = VoiceSegmentVersion(
        segment_id=seg.id,
        version_no=ver_no,
        text=seg_text,
        segment_url=seg_url,
        db=real_db,
        pitch_mean_hz=real_hz,
        rate_wpm=safe_float(wpm.get("rate_wpm", 0.0)),
        pause_ratio=0.0,  # 새로운 구조에는 없음
        prosody_score=0.0,  # 새로운 구조에는 없음
        feedback=feedback_text,
        db_list=dB_list,  # 0.1초 간격으로 측정된 dB 값 리스트
    )
    db.add(version)

    if progress_callback:
        progress_callback(90)  # DB 저장 중: 90%

    db.commit()
    db.refresh(version)

    if progress_callback:
        progress_callback(100)  # 완료: 100%

    return {
        "id": version.id,
        "segment_id": version.segment_id,
        "version_no": version.version_no,
        "text": version.text if version.text else "",
        "segment_url": version.segment_url if version.segment_url else "",
        "feedback": version.feedback if version.feedback else "",
        "dB_list": dB_list,  # 0.1초 간격으로 측정된 dB 값 리스트
        "metrics": {
            "dB": safe_float(version.db),
            "pitch_mean_hz": safe_float(version.pitch_mean_hz),
            "rate_wpm": safe_float(version.rate_wpm),
            "pause_ratio": safe_float(version.pause_ratio),
            "prosody_score": safe_float(version.prosody_score),
        }
    }
