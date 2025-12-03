# app/domain/voice/service.py
import os
import tempfile
import uuid

import librosa
from fastapi import UploadFile
from pydub import AudioSegment
from sqlalchemy.orm import Session
from typing import Optional

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.voice.utils.draw_dB_image import draw
from app.domain.llm.service.classify_text_service import classify_text_into_sections
from app.domain.voice.utils.map_sections_to_segments import (
    map_llm_sections_to_sentences_with_timestamps
)
from app.infrastructure.storage.object_storage import upload_file
from app.utils.analyzer_function import compute_energy_stats_segment, compute_final_boundary_features_for_segment, compute_pitch_cv_segment, get_voiced_mask_from_words
from app.utils.audio_analyzer import transcribe_audio, analyze_segment_audio, analyze_segments
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


def process_voice(db: Session, file: UploadFile, user: User, category_id: Optional[int], name: str):
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    object_name = f"voices/{uuid.uuid4()}"
    original_url = upload_file(tmp_path, object_name)

    clova_result = make_voice_to_stt(tmp_path)
    print(clova_result)
    full_text = clova_result["text"]
    clova_segments = clova_result.get("segments", [])
    
    # 2단계: LLM으로 문단별 분할 (part 정보를 위해)
    llm_part_map = {}  # 텍스트 -> part 매핑
    try:
        llm_sections = classify_text_into_sections(full_text)
        for item in llm_sections:
            if "sections" in item:
                for section in item["sections"]:
                    section_text = section.get("content", "").strip()
                    section_part = section.get("part", "")
                    if section_text:
                        # 텍스트의 앞부분으로 매핑
                        llm_part_map[section_text[:50]] = section_part
    except Exception as e:
        print(f"⚠️ LLM 분할 실패: {e}")
    
    # 3단계: Clova Speech segments에 part 정보 추가
    final_segments = []
    for seg in clova_segments:
        seg_text = seg.get("text", "").strip()
        
        # LLM part 정보 매칭
        part = None
        for key, value in llm_part_map.items():
            if key in seg_text or seg_text[:50] in key:
                part = value
                break
        
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
    y_audio, sr_audio = librosa.load(tmp_path, sr=16000)
    
    segments_with_metrics = []
    for seg in final_segments:
        # Clova Speech segments는 밀리초 단위이므로 초로 변환
        seg_start_sec = seg["start"] / 1000.0
        seg_end_sec = seg["end"] / 1000.0
        
        metrics = analyze_segment_audio(
            tmp_path,
            seg_start_sec,
            seg_end_sec,
            seg["text"],
            y=y_audio,
            sr=sr_audio
        )
        
        # words에 metrics 추가 (feedback 계산용)
        # words는 [start_ms, end_ms, text] 형태
        words_with_metrics = []
        for word in seg.get("words", []):
            if isinstance(word, list) and len(word) >= 3:
                word_start_ms = word[0]
                word_end_ms = word[1]
                word_text = word[2]
                word_start_sec = word_start_ms / 1000.0
                word_end_sec = word_end_ms / 1000.0
                
                word_metrics = analyze_segment_audio(
                    tmp_path,
                    word_start_sec,
                    word_end_sec,
                    y=y_audio,
                    sr=sr_audio
                )
                words_with_metrics.append({
                    "text": word_text,
                    "start": word_start_sec,
                    "end": word_end_sec,
                    "metrics": word_metrics
                })
        
        # Clova Speech segments 원본 형태 유지하면서 metrics 추가
        segments_with_metrics.append({
            "start": seg["start"],  # 밀리초 (원본 유지)
            "end": seg["end"],      # 밀리초 (원본 유지)
            "text": seg["text"],
            "confidence": seg.get("confidence", 0.0),
            "words": seg.get("words", []),  # [start_ms, end_ms, text] 형태 (원본 유지)
            "textEdited": seg.get("textEdited", seg["text"]),
            "part": seg.get("part"),
            "metrics": metrics,
            "words_with_metrics": words_with_metrics  # 분석용 words (초 단위)
        })
    
    segments_to_save = segments_with_metrics

    waveform_image = draw(tmp_path)

    # 🔹 DB 저장
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

    saved_segments = save_segments_to_storage(tmp_path, voice.id, segments_to_save, db, voice, ext)
    db.commit()

    return {
        "voice_id": voice.id,
        "original_url": voice.original_url,
        "waveform_image": waveform_image,
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

def process_voice2(db: Session, file: UploadFile, user: User, category_id: Optional[int], name: str):
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    object_name = f"voices/{uuid.uuid4()}"
    original_url = upload_file(tmp_path, object_name)

    clova_result = make_voice_to_stt(tmp_path)
    print(clova_result)
    full_text = clova_result["text"]
    clova_segments = clova_result.get("segments", [])
    
    # 2단계: LLM으로 문단별 분할 (part 정보를 위해)
    llm_part_map = {}  # 텍스트 -> part 매핑
    try:
        llm_sections = classify_text_into_sections(full_text)
        for item in llm_sections:
            if "sections" in item:
                for section in item["sections"]:
                    section_text = section.get("content", "").strip()
                    section_part = section.get("part", "")
                    if section_text:
                        # 텍스트의 앞부분으로 매핑
                        llm_part_map[section_text[:50]] = section_part
    except Exception as e:
        print(f"⚠️ LLM 분할 실패: {e}")
    
    # 3단계: Clova Speech segments에 part 정보 추가
    final_segments = []
    for seg in clova_segments:
        seg_text = seg.get("text", "").strip()
        
        # LLM part 정보 매칭
        part = None
        for key, value in llm_part_map.items():
            if key in seg_text or seg_text[:50] in key:
                part = value
                break
        
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

    print(final_segments)
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


    print(final_segments)
    # STT 결과 기반 유성 마스크 생성
    full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, final_segments)
    
    print("full_voice_masked----------")
    print(full_voice_masked)
    print("full_voice_masked----------")

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
                db = float(np.mean(librosa.amplitude_to_db(w_rms, ref=1.0)))

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
                        "dB": round(db, 2),
                        "pitch_mean_hz": round(pitch_mean, 2),
                        "pitch_std_hz": round(pitch_std, 2),
                        "duration_sec": round(duration, 3)
                    }
                })

        analyzed.append(segment_info)

    feedback = make_feedback_service.make_feedback(analyzed)
    print("feedback----------")
    print(feedback)
    print("feedback----------")