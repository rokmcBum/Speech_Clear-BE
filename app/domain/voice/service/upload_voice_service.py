# app/domain/voice/service.py
import os
import tempfile
import uuid

import librosa
from fastapi import UploadFile
from pydub import AudioSegment
from sqlalchemy.orm import Session

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment
from app.domain.voice.utils.draw_dB_image import draw
from app.domain.llm.service.classify_text_service import classify_text_into_sections
from app.domain.voice.utils.map_sections_to_segments import (
    map_llm_sections_to_whisper_segments,
    map_sentences_to_whisper_segments
)
from app.infrastructure.storage.object_storage import upload_file
from app.utils.audio_analyzer import transcribe_audio, analyze_segment_audio, analyze_segments
from app.utils.feedback_rules import make_feedback


def _remove_duplicate_segments(segments: list) -> list:
    """
    중복된 segments를 제거합니다.
    같은 시간 범위(start, end)와 텍스트를 가진 segment는 하나만 유지합니다.
    part 정보는 첫 번째로 나타난 것을 사용합니다.
    
    Args:
        segments: segment 리스트
    
    Returns:
        중복이 제거된 segment 리스트
    """
    seen = {}  # (start, end, text) -> segment
    result = []
    
    for seg in segments:
        # 시간과 텍스트로 고유 키 생성
        key = (
            round(seg.get("start", 0), 3),  # 소수점 3자리까지 반올림
            round(seg.get("end", 0), 3),
            seg.get("text", "").strip()
        )
        
        if key not in seen:
            seen[key] = seg
            result.append(seg)
        else:
            # 이미 존재하는 경우, part 정보가 있으면 업데이트 (우선순위: 서론 > 본론 > 결론)
            existing = seen[key]
            existing_part = existing.get("part", "")
            new_part = seg.get("part", "")
            
            # part 우선순위: 서론 > 본론 > 결론
            part_priority = {"서론": 1, "본론": 2, "결론": 3}
            existing_priority = part_priority.get(existing_part, 99)
            new_priority = part_priority.get(new_part, 99)
            
            # 우선순위가 높은(숫자가 작은) part로 업데이트
            if new_priority < existing_priority:
                existing["part"] = new_part
                print(f"⚠️ 중복 segment 발견: '{seg.get('text', '')[:30]}...' - part를 '{new_part}'로 업데이트")
    
    return result


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


def process_voice(db: Session, file: UploadFile, user: User, category_id: int):
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    object_name = f"voices/{uuid.uuid4()}"
    original_url = upload_file(tmp_path, object_name)

    # 1단계: Whisper로 텍스트 추출만 (문장별로 분리됨)
    whisper_result = transcribe_audio(tmp_path, model_name="turbo", language="ko")
    full_text = whisper_result["text"]
    
    # 2단계: LLM으로 문단별 분할
    try:
        llm_sections = classify_text_into_sections(full_text)
        # LLM 분할 결과를 Whisper segments와 매핑하여 시간 정보 추가 (문단 단위)
        mapped_paragraphs = map_llm_sections_to_whisper_segments(
            llm_sections, 
            whisper_result["segments"]
        )
        
        # 각 문단 범위 내의 Whisper segments를 문장 단위로 변환
        # 각 Whisper segment는 한 번만 사용되도록 보장
        used_segment_indices = set()  # 이미 사용된 Whisper segment 인덱스
        sentence_segments = []
        
        if mapped_paragraphs:
            for paragraph in mapped_paragraphs:
                # 문단 시간 범위 내의 Whisper segments를 문장 단위 segment로 변환
                paragraph_sentences = map_sentences_to_whisper_segments(
                    paragraph["text"],
                    paragraph.get("part", ""),
                    paragraph["start"],
                    paragraph["end"],
                    whisper_result["segments"],
                    used_segment_indices  # 사용된 인덱스 전달
                )
                sentence_segments.extend(paragraph_sentences)
        
        # 문장 단위 segments가 있으면 사용, 없으면 원본 Whisper segments 사용
        if sentence_segments:
            # 추가 중복 제거 (안전장치)
            final_segments = _remove_duplicate_segments(sentence_segments)
        else:
            # LLM 분할 실패 시 원본 Whisper segments 사용
            print("LLM 분할 실패 또는 문장 분할 실패, 원본 Whisper segments 사용")
            final_segments = whisper_result["segments"]
    except Exception as e:
        # LLM 분할 실패 시 원본 Whisper segments 사용
        print(f"LLM 분할 실패, 원본 segments 사용: {e}")
        final_segments = whisper_result["segments"]
    
    # 3단계: 나뉜 문장별로 librosa로 분석하여 metrics 계산
    # 오디오를 한 번만 로드하여 성능 최적화
    y_audio, sr_audio = librosa.load(tmp_path, sr=16000)
    
    segments_with_metrics = []
    for seg in final_segments:
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
    
    segments_to_save = segments_with_metrics

    waveform_image = draw(tmp_path)

    # 🔹 DB 저장
    voice = Voice(
        user_id=user.id,
        category_id=category_id,
        filename=file.filename,
        content_type=file.content_type,
        original_url=original_url,
        duration_sec=whisper_result.get("duration")
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
