# app/domain/voice/service.py
import os
import tempfile
import uuid
from typing import Optional

import librosa
from fastapi import UploadFile
from pydub import AudioSegment
from sqlalchemy.orm import Session

from app.domain.llm.service.classify_text_service import classify_text_into_sections
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.voice.utils.draw_dB_image import draw
from app.domain.voice.utils.map_sections_to_segments import (
    map_llm_sections_to_sentences_with_timestamps,
)
from app.infrastructure.storage.object_storage import upload_file
from app.utils.audio_analyzer import (
    analyze_segment_audio,
    analyze_segments,
    transcribe_audio,
)
from app.utils.feedback_rules import make_feedback


def save_segments_to_storage(local_path, voice_id, segments, db, voice, ext):
    audio = AudioSegment.from_file(local_path)
    saved_segments = []
    for order_no, seg in enumerate(segments, start=1):
        seg_audio = audio[int(seg["start"] * 1000):int(seg["end"] * 1000)]
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
        words = seg.get("words", [])
        segment = VoiceSegment(
            voice_id=voice.id,
            order_no=order_no,
            text=seg["text"],
            part=seg.get("part"), 
            start_time=float(seg["start"]),
            end_time=float(seg["end"]),
            segment_url=seg_url,
            db=float(met.get("dB", 0.0)),
            pitch_mean_hz=float(met.get("pitch_mean_hz", 0.0)),
            rate_wpm=float(met.get("rate_wpm", 0.0)),
            pause_ratio=float(met.get("pause_ratio", 0.0)),
            prosody_score=float(met.get("prosody_score", 0.0)),
            feedback=make_feedback(words),
        )
        db.add(segment)
        saved_segments.append(segment)
    return saved_segments


def process_voice(db: Session, file: UploadFile, user: User, category_id: Optional[int], name: str, progress_callback=None):
    if progress_callback:
        progress_callback(10) # 시작: 10%
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    object_name = f"voices/{uuid.uuid4()}"
    original_url = upload_file(tmp_path, object_name)

    # 1단계: Whisper로 전체 텍스트 + word timestamps 추출 (segments는 신뢰하지 않음)
    whisper_result = transcribe_audio(tmp_path, model_name="tiny", language="ko")
    full_text = whisper_result["text"]
    word_timestamps = whisper_result.get("words", [])
    
    if progress_callback:
        progress_callback(30) # Whisper 완료: 30%
    
    # 2단계: LLM으로 문단별 분할
    try:
        llm_sections = classify_text_into_sections(full_text)
        
        # 3단계: 각 문단을 kss로 문장 단위로 나누고, word timestamps로 정확한 시간 계산
        final_segments = map_llm_sections_to_sentences_with_timestamps(
            llm_sections,
            full_text,
            word_timestamps
        )
        
        if not final_segments:
            print("⚠️ LLM 분할 또는 문장 분할 실패, 빈 결과 반환")
            final_segments = []
    except Exception as e:
        # LLM 분할 실패 시 빈 결과
        print(f"⚠️ LLM 분할 실패: {e}")
        final_segments = []
    
    if progress_callback:
        progress_callback(50) # LLM 완료: 50%
    
    # 3단계: 나뉜 문장별로 librosa로 분석하여 metrics 계산
    # 오디오를 한 번만 로드하여 성능 최적화
    y_audio, sr_audio = librosa.load(tmp_path, sr=16000)
    
    segments_with_metrics = []
    for i, seg in enumerate(final_segments):
        metrics = analyze_segment_audio(
            tmp_path,
            seg["start"],
            seg["end"],
            seg["text"],
            y=y_audio,
            sr=sr_audio
        )
        
        # words에 metrics 추가 (feedback 계산용)
        words_with_metrics = []
        for word in seg.get("words", []):
            word_metrics = analyze_segment_audio(
                tmp_path,
                word["start"],
                word["end"],
                y=y_audio,
                sr=sr_audio
            )
            words_with_metrics.append({
                "text": word["text"],
                "start": word["start"],
                "end": word["end"],
                "metrics": word_metrics
            })
        
        segments_with_metrics.append({
            "text": seg["text"],
            "part": seg.get("part"),
            "start": seg["start"],
            "end": seg["end"],
            "metrics": metrics,
            "words": words_with_metrics
        })
        
        # 진행률 업데이트 (50% ~ 90% 사이)
        if progress_callback:
            current_progress = 50 + int((i + 1) / len(final_segments) * 40)
            progress_callback(current_progress)
    
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
        duration_sec=whisper_result.get("duration")
    )
    db.add(voice)
    db.flush()

    saved_segments = save_segments_to_storage(tmp_path, voice.id, segments_to_save, db, voice, ext)
    db.commit()

    if progress_callback:
        progress_callback(100) # 완료: 100%

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
