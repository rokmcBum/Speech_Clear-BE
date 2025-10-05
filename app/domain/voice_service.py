# app/domain/voice/service.py
import os, tempfile, uuid
from fastapi import UploadFile
from sqlalchemy.orm import Session
from pydub import AudioSegment

from app.domain.voice.model.voice import Voice, VoiceSegment
from app.infrastructure.storage.object_storage import upload_file
from app.utils.audio_analyzer import analyze_segments
from app.utils.feedback_rules import make_feedback


def save_segments_to_storage(local_path, voice_id, segments, db, voice, ext):
    audio = AudioSegment.from_file(local_path)
    saved_segments = []

    for order_no, seg in enumerate(segments, start=1):
        seg_audio = audio[int(seg["start"]*1000):int(seg["end"]*1000)]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        ext = ext.replace(".", "")
        format_map = {
            "m4a": "mp4",   # 👈 m4a는 mp4로 매핑
            "aac": "adts",
            "wav": "wav",
            "mp3": "mp3"
        }
        fmt = format_map.get(ext, ext)  # 기본은 그대로
        seg_audio.export(tmp_file.name, format=fmt)

        object_name = f"voices/{voice_id}/segments/seg_{order_no}{ext}"
        seg_url = upload_file(tmp_file.name, object_name)

        met = seg.get("metrics", {})
        segment = VoiceSegment(
            voice_id=voice.id,
            order_no=order_no,
            text=seg["text"],
            start_time=float(seg["start"]),
            end_time=float(seg["end"]),
            segment_url=seg_url,
            db=float(met.get("dB")),
            pitch_mean_hz=float(met.get("pitch_mean_hz")),
            rate_wpm=float(met.get("rate_wpm")),
            pause_ratio=float(met.get("pause_ratio")),
            prosody_score=float(met.get("prosody_score")),
            feedback=make_feedback(met),
        )
        db.add(segment)
        saved_segments.append(segment)
    return saved_segments


def process_voice(db: Session, file: UploadFile):
    # 1. 임시 저장
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # 2. 원본 업로드
    object_name = f"voices/{uuid.uuid4()}{ext}"
    original_url = upload_file(tmp_path, object_name)

    # 3. 분석
    analysis = analyze_segments(tmp_path, model_name="turbo", language="ko")

    # 4. DB 저장
    voice = Voice(
        filename=file.filename,
        content_type=file.content_type,
        original_url=original_url,
        duration_sec=analysis.get("duration")
    )
    db.add(voice)
    db.flush()

    # 5. segment 저장
    saved_segments = save_segments_to_storage(tmp_path, voice.id, analysis["segments"], db, voice, ext)
    db.commit()

    # 6. 응답
    return {
        "voice_id": voice.id,
        "original_url": voice.original_url,
        "segments": [
            {
                "id": seg.id,
                "order_no": seg.order_no,
                "text": seg.text,
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
