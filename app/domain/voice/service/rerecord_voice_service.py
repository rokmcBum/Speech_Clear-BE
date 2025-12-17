import json
import os
import tempfile
from typing import Callable, Optional

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette import status

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.make_feedback_service import make_re_recording_feedback
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment, VoiceSegmentVersion
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import upload_file
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
    make_part_index_map,
)


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


def re_record_segment(
    db: Session,
    file: UploadFile,
    segment_id: int,
    user: User,
    db_list_str: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    file_content: bytes = None
):
    if progress_callback:
        progress_callback(10)  # 시작: 10%
    
    if progress_callback:
        progress_callback(10)  # 시작: 10%
    
    try:
        seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
        
        if not seg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
        
        # voice 소유권 검증
        voice = verify_voice_ownership(seg.voice_id, user, db)
        print(f"[DEBUG] Re-record: Voice ownership verified. ID={voice.id}")

        ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
            if file_content:
                tmp.write(file_content)
            else:
                tmp.write(file.file.read())
            tmp_path = tmp.name
        print(f"[DEBUG] Re-record: Temp file created at {tmp_path} (Size: {os.path.getsize(tmp_path)})")

        if progress_callback:
            progress_callback(20)  # 파일 저장 완료: 20%

        ver_no = _next_version_no(db, segment_id)
        object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
        seg_url = upload_file(tmp_path, object_name)
        print(f"[DEBUG] Re-record: File uploaded to {object_name}")

        if progress_callback:
            progress_callback(30)  # Object Storage 업로드 완료: 30%

        # Clova Speech API로 STT 수행
        print("[DEBUG] Re-record: Calling Clova STT")
        clova_result = make_voice_to_stt(tmp_path)
        print("[DEBUG] Re-record: Clova STT Done")
        
        if progress_callback:
            progress_callback(40)  # Clova Speech 완료: 40%
        
        full_text = clova_result["text"]
        clova_words = clova_result.get("words", [])  # Clova Speech의 word timestamps
        
        # librosa로 오디오 분석
        print("[DEBUG] Re-record: Calling Librosa load")
        y, sr = librosa.load(tmp_path, sr=16000)
        print(f"[DEBUG] Re-record: Librosa loaded. sr={sr}, shape={y.shape}")
        
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
        # seg_words 형식 변환: segments 내부 words는 [start_ms, end_ms, text] 형식
        words_for_mask = []
        for w in seg_words:
            if isinstance(w, list) and len(w) >= 3:
                # 리스트 형식 [start_ms, end_ms, text]
                words_for_mask.append([w[0], w[1], w[2]])
            elif isinstance(w, dict):
                # 딕셔너리 형식 {"text": ..., "start": ..., "end": ...} (초 단위)
                words_for_mask.append([
                    int(w.get("start", 0.0) * 1000),  # 밀리초로 변환
                    int(w.get("end", 0.0) * 1000),    # 밀리초로 변환
                    w.get("text", "")
                ])
        
        final_segments_for_mask = [{
            "words": words_for_mask
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
        
        # dB calculation: RMS -> dB (ref=1.0)
        mean_db = 0.0
        if len(rms_seg) > 0:
            mean_db = float(np.mean(librosa.amplitude_to_db(rms_seg, ref=1.0)))

        # pitch 계산
        mean_st, std_st, cv_pitch = compute_pitch_cv_segment(
            f0_hz=f0_seg,
            f0_min=1e-3
        )
        
        # mean_hz 계산
        mean_hz = 0.0
        if len(f0_seg) > 0:
            valid_f0 = f0_seg[f0_seg > 1e-3]
            if len(valid_f0) > 0:
                mean_hz = float(np.mean(valid_f0))
        
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
                "mean_db": safe_float(mean_db)
            },
            "pitch": {
                "mean_st": safe_float(mean_st),
                "std_st": safe_float(std_st),
                "cv": safe_float(cv_pitch),
                "mean_hz": safe_float(mean_hz)
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
            # words 형식 처리: 리스트 [start_ms, end_ms, text] 또는 딕셔너리 {"text": ..., "start": ..., "end": ...}
            if isinstance(w, list) and len(w) >= 3:
                w_text = w[2].strip()
                w_start, w_end = w[0]/1000, w[1]/1000
            elif isinstance(w, dict):
                w_text = w.get("text", "").strip()
                w_start = w.get("start", 0.0)  # 초 단위
                w_end = w.get("end", 0.0)      # 초 단위
            else:
                continue
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

        if progress_callback:
            progress_callback(70)  # 재녹음 오디오 분석 완료: 70%

        # 저장된 원본 voice의 sentence_feedback 사용
        feedback_text = ""
        re_analyzed_metrics = None
        
        try:
            # 이전 재녹음 분석 결과가 있으면 그것을 원본으로 사용, 없으면 원본 voice의 sentence_feedback 사용
            last_re_analyzed_metrics = seg.last_re_analyzed_metrics
            sentence_feedback = voice.sentence_feedback
            
            if last_re_analyzed_metrics:
                # 이전 재녹음 결과가 있으면 그것을 원본으로 사용
                # sentence_feedback 형식으로 변환 (해당 segment만)
                segment_order_no = seg.order_no - 1
                original_analyzed_metrics = last_re_analyzed_metrics
                
                # sentence_feedback 형식으로 변환
                modified_sentence_feedback = []
                if sentence_feedback:
                    # 원본 sentence_feedback 복사
                    modified_sentence_feedback = [s.copy() for s in sentence_feedback]
                    # 해당 segment의 analyzed를 이전 재녹음 결과로 교체
                    for s in modified_sentence_feedback:
                        if s.get("id") == segment_order_no:
                            s["analyzed"] = original_analyzed_metrics
                            break
                else:
                    # sentence_feedback이 없으면 새로 생성
                    modified_sentence_feedback = [{
                        "id": segment_order_no,
                        "analyzed": original_analyzed_metrics
                    }]
                
                if progress_callback:
                    progress_callback(75)  # 이전 재녹음 결과 조회 완료: 75%
                
                # 재녹음 피드백 생성 (이전 재녹음 결과를 원본으로 사용)
                feedback_text, re_analyzed_metrics = make_re_recording_feedback(
                    sentence_feedback=modified_sentence_feedback,
                    id=segment_order_no,
                    re_recording_path=tmp_path
                )
            elif sentence_feedback:
                # 원본 voice의 sentence_feedback 사용
                if progress_callback:
                    progress_callback(75)  # 원본 sentence_feedback 조회 완료: 75%
                
                # 재녹음 피드백 생성 (make_re_recording_feedback 사용)
                segment_order_no = seg.order_no - 1  # id는 0부터 시작
                feedback_text, re_analyzed_metrics = make_re_recording_feedback(
                    sentence_feedback=sentence_feedback,
                    id=segment_order_no,
                    re_recording_path=tmp_path
                )
            else:
                # sentence_feedback이 없는 경우 (기존 데이터) 기본 피드백 생성
                print(f"[WARN] voice_id={voice.id}에 sentence_feedback이 없습니다. 기본 피드백을 생성합니다.")
                feedbacks_list, _ = make_feedback_service.make_feedback([segment_info])
                if feedbacks_list and len(feedbacks_list) > 0:
                    feedback_text = feedbacks_list[0].get("feedback", "")
                else:
                    feedback_text = "재녹음 분석이 완료되었습니다."
                re_analyzed_metrics = None
            
            if progress_callback:
                progress_callback(85)  # 재녹음 피드백 생성 완료: 85%
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 재녹음 피드백 생성 실패: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # 에러 발생 시 기본 피드백 생성
            feedbacks_list, _ = make_feedback_service.make_feedback([segment_info])
            if feedbacks_list and len(feedbacks_list) > 0:
                feedback_text = feedbacks_list[0].get("feedback", "")
            else:
                feedback_text = "재녹음 피드백 분석에 오류가 발생하였습니다."

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

        # DB에 저장할 메트릭 계산
        energy = segment_info.get("energy", {})
        pitch = segment_info.get("pitch", {})
        wpm = segment_info.get("wpm", {})
        
        # real_db 계산
        real_db = 0.0
        real_hz = 0.0
        
        try:
            y_seg_audio, sr_seg = librosa.load(tmp_path, sr=16000)
            if len(y_seg_audio) > 0:
                rms_full = librosa.feature.rms(y=y_seg_audio)
                real_db = float(np.mean(librosa.amplitude_to_db(rms_full, ref=1.0)))
        except Exception as e:
            print(f"⚠️ 세그먼트 오디오 재로드 실패: {e}")
            real_db = safe_float(energy.get("mean_db", 0.0))
        
        real_hz = safe_float(pitch.get("mean_hz", 0.0))
        
        real_db = safe_float(real_db)
        real_hz = safe_float(real_hz)

        # 임시 파일 정리
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f"⚠️ 임시 파일 삭제 실패: {e}")

        version = VoiceSegmentVersion(
            segment_id=seg.id,
            version_no=ver_no,
            text=seg_text,
            segment_url=seg_url,
            db=real_db,
            pitch_mean_hz=real_hz,
            rate_wpm=safe_float(wpm.get("rate_wpm", 0.0)),
            pause_ratio=0.0,
            prosody_score=0.0,
            feedback=feedback_text,
            db_list=dB_list,
        )
        db.add(version)
        
        if 're_analyzed_metrics' in locals() and re_analyzed_metrics:
            seg.last_re_analyzed_metrics = re_analyzed_metrics

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
            "dB_list": dB_list,
            "metrics": {
                "dB": safe_float(version.db),
                "pitch_mean_hz": safe_float(version.pitch_mean_hz),
                "rate_wpm": safe_float(version.rate_wpm),
                "pause_ratio": safe_float(version.pause_ratio),
                "prosody_score": safe_float(version.prosody_score),
            }
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[CRITICAL ERROR] 재녹음 처리 중 예외 발생: {e}")
        print(traceback_str)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
