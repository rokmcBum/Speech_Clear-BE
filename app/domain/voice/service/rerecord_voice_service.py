import os
import tempfile

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette import status

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment, VoiceSegmentVersion
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import upload_file, download_file
from app.domain.llm.service.make_feedback_service import make_re_recording_feedback, make_feedback
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
    make_part_index_map
)
import librosa
import numpy as np


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


def re_record_segment(db: Session, segment_id: int, file: UploadFile, user: User):
    seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
    
    if not seg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    
    # voice 소유권 검증
    voice = verify_voice_ownership(seg.voice_id, user, db)

    ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    ver_no = _next_version_no(db, segment_id)
    object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
    seg_url = upload_file(tmp_path, object_name)

    # 원본 voice의 모든 세그먼트 가져오기 (order_no 순서대로)
    original_segments = (
        db.query(VoiceSegment)
        .filter(VoiceSegment.voice_id == seg.voice_id)
        .order_by(VoiceSegment.order_no.asc())
        .all()
    )

    # 원본 voice의 오디오 파일 다운로드
    original_audio_path = None
    try:
        original_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a").name
        download_file(voice.original_url, original_audio_path)
        
        # 원본 오디오 분석 (전체 분석 결과 재구성)
        clova_result = make_voice_to_stt(original_audio_path)
        final_segments = []
        for clova_seg in clova_result.get("segments", []):
            words = []
            # words는 [start_ms, end_ms, text] 형태의 리스트이거나 {"start": ..., "end": ..., "text": ...} 형태의 딕셔너리일 수 있음
            for w in clova_seg.get("words", []):
                if isinstance(w, list) and len(w) >= 3:
                    # 리스트 형태: [start_ms, end_ms, text]
                    words.append([int(w[0]), int(w[1]), w[2]])
                elif isinstance(w, dict):
                    # 딕셔너리 형태: {"start": ..., "end": ..., "text": ...}
                    start_ms = int(w.get("start", 0) * 1000) if isinstance(w.get("start"), (int, float)) else int(w.get("start", 0))
                    end_ms = int(w.get("end", 0) * 1000) if isinstance(w.get("end"), (int, float)) else int(w.get("end", 0))
                    words.append([start_ms, end_ms, w.get("text", "")])
                else:
                    print(f"[WARN] 알 수 없는 word 형식: {w}")
                    continue
            
            # segment의 start, end는 초 단위이거나 밀리초 단위일 수 있음
            seg_start = clova_seg.get("start", 0)
            seg_end = clova_seg.get("end", 0)
            
            # 초 단위인 경우 밀리초로 변환
            if isinstance(seg_start, (int, float)) and seg_start < 1000:  # 1000초 미만이면 초 단위로 가정
                seg_start_ms = int(seg_start * 1000)
            else:
                seg_start_ms = int(seg_start)
                
            if isinstance(seg_end, (int, float)) and seg_end < 1000:  # 1000초 미만이면 초 단위로 가정
                seg_end_ms = int(seg_end * 1000)
            else:
                seg_end_ms = int(seg_end)
            
            final_segments.append({
                "start": seg_start_ms,
                "end": seg_end_ms,
                "text": clova_seg.get("text", ""),
                "words": words,
                "part": None  # 원본 세그먼트에서 part 정보 가져오기
            })
        
        # 원본 세그먼트의 part 정보 매핑
        for i, orig_seg in enumerate(original_segments):
            if i < len(final_segments):
                final_segments[i]["part"] = orig_seg.part

        # 원본 오디오 분석 (librosa)
        y, sr = librosa.load(original_audio_path, sr=16000)
        frame_length = 2048
        hop_length = 256

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=300, frame_length=frame_length, hop_length=hop_length)

        frame_idx = np.arange(rms.shape[0])
        frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr

        full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, final_segments)

        # 원본 세그먼트 분석 결과 재구성
        analyzed = []
        for idx, seg_item in enumerate(final_segments):
            seg_start, seg_end = seg_item["start"]/1000, seg_item["end"]/1000
            seg_text = seg_item["text"].strip()

            y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)
            seg_voice_masked = y_seg & full_voice_masked

            rms_seg = rms[seg_voice_masked]
            f0_seg = f0[seg_voice_masked]

            mean_r, std_r, cv_energy = compute_energy_stats_segment(rms=rms_seg, silence_thresh=1e-6)
            mean_st, std_st, cv_pitch = compute_pitch_cv_segment(f0_hz=f0_seg, f0_min=1e-3)

            seg_frame_idx = np.arange(rms[y_seg].shape[0])
            seg_frame_times = (seg_frame_idx * hop_length + hop_length / 2.0) / sr

            final_rms_ratio, final_rms_slope, final_pitch_semitone_drop, final_pitch_semitone_slope = compute_final_boundary_features_for_segment(
                rms=rms[y_seg],
                f0_hz=f0[y_seg],
                voice_masked=full_voice_masked[y_seg],
                frame_times=seg_frame_times,
                seg_length=seg_end-seg_start
            )

            words_count = len(seg_text.split())
            duration_min = (seg_end - seg_start) / 60
            rate_wpm = words_count / duration_min if duration_min > 0 else 0

            segment_info = {
                "id": idx,
                "part": seg_item.get("part"),
                "text": seg_text,
                "start": seg_start,
                "end": seg_end,
                "energy": {
                    "mean_rms": round(mean_r, 2),
                    "std_rms": round(std_r, 2),
                    "cv": round(cv_energy, 4)
                },
                "pitch": {
                    "mean_st": round(mean_st, 2),
                    "std_st": round(std_st, 2),
                    "cv": round(cv_pitch, 4)
                },
                "wpm": {
                    "word_count": words_count,
                    "rate_wpm": round(rate_wpm, 1),
                    "duration_sec": round(seg_end - seg_start, 3)
                },
                "final_boundary": {
                    "final_rms_ratio": round(final_rms_ratio, 2) if not np.isnan(final_rms_ratio) else "NaN",
                    "final_rms_slope": round(final_rms_slope, 4) if not np.isnan(final_rms_slope) else "NaN",
                    "final_pitch_semitone_drop": round(final_pitch_semitone_drop, 2) if not np.isnan(final_pitch_semitone_drop) else "NaN",
                    "final_pitch_semitone_slope": round(final_pitch_semitone_slope, 4) if not np.isnan(final_pitch_semitone_slope) else "NaN"
                },
                "words": []
            }
            analyzed.append(segment_info)

        # 원본 voice의 sentence_feedback 생성
        paragraph_index = make_part_index_map(final_segments)
        sentence_feedback, _ = make_feedback(analyzed, paragraph_index)
        
        print(f"[DEBUG] sentence_feedback 생성 완료: {len(sentence_feedback)}개")
        print(f"[DEBUG] 재녹음할 segment order_no: {seg.order_no}, id: {seg.order_no - 1}")

        # 재녹음 피드백 생성 (make_re_recording_feedback 사용)
        segment_order_no = seg.order_no - 1  # id는 0부터 시작
        feedback, re_analyzed_metrics = make_re_recording_feedback(
            sentence_feedback=sentence_feedback,
            id=segment_order_no,
            re_recording_path=tmp_path
        )
        
        print(f"[DEBUG] 재녹음 피드백 생성 완료: feedback={feedback[:100] if feedback else 'None'}...")

    except Exception as e:
        import traceback
        print(f"[ERROR] 원본 오디오 분석 실패: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        # 원본 분석 실패 시 기본 피드백 생성
        feedback = None
        re_analyzed_metrics = {}
    finally:
        if original_audio_path and os.path.exists(original_audio_path):
            try:
                os.remove(original_audio_path)
            except:
                pass
    
    # 재녹음 오디오 분석
    y_re, sr_re = librosa.load(tmp_path, sr=16000)
    stt_result = make_voice_to_stt(tmp_path)
    text = stt_result.get("text", "").strip()
    
    # 간단한 메트릭 계산
    rms_re = librosa.feature.rms(y=y_re)[0]
    mean_db = float(np.mean(librosa.amplitude_to_db(rms_re, ref=1.0))) if len(rms_re) > 0 else 0.0
    
    f0_re, _, _ = librosa.pyin(y_re, fmin=80, fmax=300)
    pitch_vals = f0_re[~np.isnan(f0_re)]
    pitch_mean_hz = float(np.mean(pitch_vals)) if pitch_vals.size > 0 else 0.0
    
    words_count = len(text.split())
    duration_sec = len(y_re) / sr_re
    rate_wpm = (words_count / duration_sec * 60) if duration_sec > 0 else 0.0

    # feedback이 None이거나 빈 문자열인 경우 처리
    if feedback is None or (isinstance(feedback, str) and feedback.strip() == ""):
        print(f"[WARN] feedback이 비어있어 기본값으로 설정합니다.")
        feedback = "재녹음 분석이 완료되었습니다."
    
    version = VoiceSegmentVersion(
        segment_id=seg.id,
        version_no=ver_no,
        text=text,
        segment_url=seg_url,
        db=mean_db,
        pitch_mean_hz=pitch_mean_hz,
        rate_wpm=rate_wpm,
        pause_ratio=0.0,  # 재녹음은 단일 문장이므로 pause_ratio 계산 생략
        prosody_score=0.0,  # 재녹음은 단일 문장이므로 prosody_score 계산 생략
        feedback=feedback,
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
        "metrics": {
            "dB": version.db,
            "pitch_mean_hz": version.pitch_mean_hz,
            "rate_wpm": version.rate_wpm,
            "pause_ratio": version.pause_ratio,
            "prosody_score": version.prosody_score,
        }
    }
