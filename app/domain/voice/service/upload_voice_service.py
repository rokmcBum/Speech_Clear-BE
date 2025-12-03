# app/domain/voice/service.py
import os
import tempfile
import uuid
from typing import Optional

import librosa
from fastapi import UploadFile
from pydub import AudioSegment
from sqlalchemy.orm import Session

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.llm.service.classify_text_service import classify_text_into_sections
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.voice.utils.draw_dB_image import draw
from app.domain.voice.utils.map_sections_to_segments import (
    map_llm_sections_to_sentences_with_timestamps,
)
from app.infrastructure.storage.object_storage import upload_file
from app.utils.analyzer_function import compute_energy_stats_segment, compute_final_boundary_features_for_segment, compute_pitch_cv_segment, get_voiced_mask_from_words
from app.utils.audio_analyzer import transcribe_audio, analyze_segment_audio, analyze_segments
from app.utils.audio_analyzer import (
    analyze_segment_audio,
    analyze_segments,
    transcribe_audio,
)
from app.utils.feedback_rules import make_feedback

import numpy as np


def save_segments_to_storage(local_path, voice_id, segments, db, voice, ext):
    audio = AudioSegment.from_file(local_path)
    saved_segments = []
    for order_no, seg in enumerate(segments, start=1):
        # Clova Speech segments는 밀리초 단위이므로 그대로 사용
        seg_start_ms = seg["start"]  # 밀리초
        seg_end_ms = seg["end"]      # 밀리초
        seg_audio = audio[seg_start_ms:seg_end_ms]
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        ext = ext.replace(".", "")
        format_map = {
            "m4a": "mp4",
            "aac": "adts",
            "wav": "wav",
            "mp3": "mp3"
        }
        fmt = format_map.get(ext, ext)  # 기본은 그대로
        seg_audio.export(tmp_file.name, format=fmt)

        object_name = f"voices/{voice_id}/segments/seg_{order_no}"
        seg_url = upload_file(tmp_file.name, object_name)
        met = seg.get("metrics", {})
        words = seg.get("words", [])  # [start_ms, end_ms, text] 형태
        
        # DB에는 초 단위로 저장
        segment = VoiceSegment(
            voice_id=voice.id,
            order_no=order_no,
            text=seg["text"],
            part=seg.get("part"), 
            start_time=float(seg_start_ms / 1000.0),  # 밀리초를 초로 변환
            end_time=float(seg_end_ms / 1000.0),      # 밀리초를 초로 변환
            segment_url=seg_url,
            db=float(met.get("dB", 0.0)),
            pitch_mean_hz=float(met.get("pitch_mean_hz", 0.0)),
            rate_wpm=float(met.get("rate_wpm", 0.0)),
            pause_ratio=float(met.get("pause_ratio", 0.0)),
            prosody_score=float(met.get("prosody_score", 0.0)),
            # feedback=make_feedback(words),
        )
        db.add(segment)
        saved_segments.append(segment)
    return saved_segments


def process_voice(db: Session, file: UploadFile, user: User, category_id: Optional[int], name: str, progress_callback=None, file_content: bytes = None):
    if progress_callback:
        progress_callback(10) # 시작: 10%
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        if file_content:
            tmp.write(file_content)
        else:
            tmp.write(file.file.read())
        tmp_path = tmp.name

    object_name = f"voices/{uuid.uuid4()}"
    original_url = upload_file(tmp_path, object_name)

    clova_result = make_voice_to_stt(tmp_path)
    print(clova_result)
    full_text = clova_result["text"]
    clova_segments = clova_result.get("segments", [])
    
    if progress_callback:
        progress_callback(30) # Whisper 완료: 30%
    # 2단계: LLM으로 문단별 분할 (part 정보를 위해)
    llm_sections_list = []  # (section_text, part) 튜플 리스트
    try:
        llm_sections = classify_text_into_sections(full_text)
        for item in llm_sections:
            if "sections" in item:
                for section in item["sections"]:
                    section_text = section.get("content", "").strip()
                    section_part = section.get("part", "")
                    if section_text and section_part:
                        llm_sections_list.append((section_text, section_part))
    except Exception as e:
        print(f"⚠️ LLM 분할 실패: {e}")
    
    if progress_callback:
        progress_callback(50) # LLM 완료: 50%

        # 3단계: Clova Speech segments에 part 정보 추가
    final_segments = []
    for seg in clova_segments:
        seg_text = seg.get("text", "").strip()
        
        # LLM part 정보 매칭 (더 정확한 매칭)
        part = None
        if llm_sections_list:
            # 텍스트 유사도 기반 매칭 (공통 단어 수)
            best_match_score = 0
            best_match_part = None
            
            for section_text, section_part in llm_sections_list:
                # 공백 제거 후 비교
                seg_text_normalized = seg_text.replace(" ", "").replace(".", "").replace(",", "")
                section_text_normalized = section_text.replace(" ", "").replace(".", "").replace(",", "")
                
                # 부분 문자열 매칭
                if seg_text_normalized in section_text_normalized or section_text_normalized in seg_text_normalized:
                    # 매칭 길이 계산
                    match_length = min(len(seg_text_normalized), len(section_text_normalized))
                    if match_length > best_match_score:
                        best_match_score = match_length
                        best_match_part = section_part
                
                # 공통 단어 기반 매칭
                seg_words = set(seg_text.split())
                section_words = set(section_text.split())
                common_words = seg_words & section_words
                if len(common_words) >= 2:  # 최소 2개 단어 이상 공통
                    if len(common_words) > best_match_score:
                        best_match_score = len(common_words)
                        best_match_part = section_part
            
            part = best_match_part
        
        # Clova Speech segments 원본 형태 유지 (start, end는 밀리초, words는 배열)
        final_seg = {
            "start": seg.get("start", 0),  # 밀리초
            "end": seg.get("end", 0),      # 밀리초
            "text": seg_text,
            "confidence": seg.get("confidence", 0.0),
            "words": seg.get("words", []),  # [start_ms, end_ms, text] 형태
            "textEdited": seg.get("textEdited", seg_text)
        }
        
        if part:
            final_seg["part"] = part
        
        final_segments.append(final_seg)

    
    # 3단계: 나뉜 문장별로 librosa로 분석하여 metrics 계산
    # 오디오를 한 번만 로드하여 성능 최적화
    # -------------------------------------------------------
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


    # STT 결과 기반 유성 마스크 생성
    full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, final_segments)
    
    result_text = ""
    analyzed = []
    id=0

    for seg in final_segments:
        # 문장 단위 정보 (시간, 텍스트)
        seg_start, seg_end = seg["start"]/1000, seg["end"]/1000
        seg_text = seg["text"].strip()
        result_text += " " + seg_text

        # 이 문장에 속하는 프레임 인덱스
        y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)

        if len(y_seg) > 0:

            # 이 문장 구간 + 유성 마스크 둘 다 만족하는 프레임
            seg_voice_masked = y_seg & full_voice_masked

            rms_seg = rms[seg_voice_masked]
            f0_seg  = f0[seg_voice_masked]

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
            
            # 문장 끝 경계 특징 계산
            final_db_drop, final_db_slope, final_pitch_drop, final_pitch_slope = compute_final_boundary_features_for_segment(
                rms=rms,
                f0_hz=f0,
                frame_times=frame_times,
                seg_start=seg_start,
                seg_end=seg_end
            )

            # 말하기 속도 계산 (wpm)
            words_count = len(seg_text.split())
            duration_min = (seg_end - seg_start) / 60
            rate_wpm = words_count / duration_min if duration_min > 0 else 0

        # 문장 단위 정보 구성
        segment_info ={
            "id": id,
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "energy": {
                "mean_rms": round(mean_r, 2),
                "std_rms": round(std_r, 2),
                "cv": round(cv_energy, 4)
            },
            "pitch": {
                "mean_hz": round(mean_st, 2),
                "std_hz": round(std_st, 2),
                "cv": round(cv_pitch, 4)
            },
            "wpm":{
                "word_count": words_count,
                "rate_wpm": round(rate_wpm, 1),
                "duration_sec": round(seg_end - seg_start, 3)
            },
            "final_boundary": {
                "final_db_drop": round(final_db_drop, 2),
                "final_db_slope": round(final_db_slope, 4),
                "final_pitch_drop_semitone": round(final_pitch_drop, 2),
                "final_pitch_slope": round(final_pitch_slope, 4)
            },
            "words" : []
        }
        id+=1

        # 단어 단위 분석
        if "words" in seg:
            for w in seg["words"]:
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

                duration = w_end - w_start

                segment_info["words"].append({
                    "text": w_text,
                    "start": w_start,
                    "end": w_end,
                    "metrics": {
                        "dB": round(db_value, 2),
                        "pitch_mean_hz": round(pitch_mean, 2),
                        "pitch_std_hz": round(pitch_std, 2),
                        "duration_sec": round(duration, 3)
                    }
                })
        if progress_callback:
            current_progress = 50 + int((id + 1) / len(final_segments) * 40)
            progress_callback(current_progress)  
        analyzed.append(segment_info)

    feedbacks_list = make_feedback_service.make_feedback(analyzed)

    # 피드백을 segment_index로 매핑
    feedback_map = {fb["segment_index"]: fb["feedback"] for fb in feedbacks_list}
        # 진행률 업데이트 (50% ~ 90% 사이)
    # Voice 생성 및 저장
    voice = Voice(
        user_id=user.id,
        category_id=category_id,
        name=name,
        filename=file.filename,
        content_type=file.content_type,
        original_url=original_url,
        duration_sec=clova_result.get("duration")
    )
    db.add(voice)
    db.flush()
    
    # 세그먼트 저장 (object storage + DB)
    audio = AudioSegment.from_file(tmp_path)
    saved_segments = []
    
    for order_no, (seg, analyzed_seg) in enumerate(zip(final_segments, analyzed), start=1):
        # Clova Speech segments는 밀리초 단위
        seg_start_ms = seg["start"]  # 밀리초
        seg_end_ms = seg["end"]      # 밀리초
        
        # Object storage에 세그먼트 저장
        seg_audio = audio[seg_start_ms:seg_end_ms]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        ext_clean = ext.replace(".", "")
        format_map = {
            "m4a": "mp4",
            "aac": "adts",
            "wav": "wav",
            "mp3": "mp3"
        }
        fmt = format_map.get(ext_clean, ext_clean)
        seg_audio.export(tmp_file.name, format=fmt)
        
        object_name = f"voices/{voice.id}/segments/seg_{order_no}"
        seg_url = upload_file(tmp_file.name, object_name)
        
        # 피드백 가져오기 (segment_index는 0부터 시작, order_no는 1부터 시작)
        segment_index = order_no - 1
        feedback_text = feedback_map.get(segment_index, "")
        
        # DB에 저장 (analyzed_seg의 정보 사용)
        energy = analyzed_seg.get("energy", {})
        pitch = analyzed_seg.get("pitch", {})
        wpm = analyzed_seg.get("wpm", {})
        
        # 기존 DB 구조에 맞게 변환
        segment = VoiceSegment(
            voice_id=voice.id,
            order_no=order_no,
            text=seg["text"],
            part=seg.get("part"),
            start_time=float(analyzed_seg["start"]),  # 초 단위
            end_time=float(analyzed_seg["end"]),      # 초 단위
            segment_url=seg_url,
            db=float(energy.get("mean_rms", 0.0)),  # mean_rms를 dB로 사용
            pitch_mean_hz=float(pitch.get("mean_hz", 0.0)),
            rate_wpm=float(wpm.get("rate_wpm", 0.0)),
            pause_ratio=0.0,  # 새로운 구조에는 없음
            prosody_score=0.0,  # 새로운 구조에는 없음
            feedback=feedback_text,
        )
        db.add(segment)
        saved_segments.append(segment)
    
    db.commit()

    if progress_callback:
        progress_callback(100) # 완료: 100%

    return {
        "voice_id": voice.id,
        "original_url": voice.original_url,
        "segments": [
            {
                "id": seg.id,
                "order_no": seg.order_no,
                "text": seg.text,
                "part": seg.part,
                "start": seg.start_time,
                "end": seg.end_time,
                "segment_url": seg.segment_url,
                "feedback": seg.feedback,
                "metrics": {
                    "dB": seg.db,
                    "pitch_mean_hz": seg.pitch_mean_hz,
                    "rate_wpm": seg.rate_wpm,
                    "pause_ratio": seg.pause_ratio,
                    "prosody_score": seg.prosody_score,
                }
            } for seg in saved_segments
        ]
    }