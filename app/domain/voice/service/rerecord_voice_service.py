import os
import tempfile

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.domain.voice.model.voice import VoiceSegment, VoiceSegmentVersion
from app.domain.voice.service.draw_dB_image_service import draw
from app.infrastructure.storage.object_storage import upload_file
from app.utils.audio_analyzer import analyze_segments
from app.utils.feedback_rules import make_feedback


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


def re_record_segment(db: Session, segment_id: int, file: UploadFile):
    seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
    if not seg:
        return None

    ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    ver_no = _next_version_no(db, segment_id)
    object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
    seg_url = upload_file(tmp_path, object_name)

    analysis = analyze_segments(tmp_path, model_name="turbo", language="ko")
    waveform_image = draw(tmp_path)
    met = {}
    word_metrics = []
    text = analysis.get("text", "").strip()

    if analysis.get("segments"):
        seg0 = analysis["segments"][0]
        met = seg0.get("metrics", {})
        word_metrics = seg0.get("words", [])

    feedback = make_feedback(word_metrics)

    version = VoiceSegmentVersion(
        segment_id=seg.id,
        version_no=ver_no,
        text=text,
        segment_url=seg_url,
        db=_to_float(met.get("dB")),
        pitch_mean_hz=_to_float(met.get("pitch_mean_hz")),
        rate_wpm=_to_float(met.get("rate_wpm")),
        pause_ratio=_to_float(met.get("pause_ratio")),
        prosody_score=_to_float(met.get("prosody_score")),
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
        "waveform_image": waveform_image,
        "metrics": {
            "dB": version.db,
            "pitch_mean_hz": version.pitch_mean_hz,
            "rate_wpm": version.rate_wpm,
            "pause_ratio": version.pause_ratio,
            "prosody_score": version.prosody_score,
        }
    }
