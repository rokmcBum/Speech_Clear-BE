import os, tempfile, uuid
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.domain.voice.model.voice import VoiceSegment, VoiceSegmentVersion
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
    # 1) 원본 segment 조회
    seg = db.query(VoiceSegment).filter(VoiceSegment.id == segment_id).first()
    if not seg:
        return None

    # 2) 업로드 파일 임시 저장
    ext_with_dot = os.path.splitext(file.filename)[1]  # ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_with_dot) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # 3) Object Storage 업로드 (버전별 디렉토리 분리 권장)
    ver_no = _next_version_no(db, segment_id)
    object_name = f"voices/{seg.voice_id}/segments/{seg.id}/v{ver_no}"
    seg_url = upload_file(tmp_path, object_name)

    # 4) 분석 (새 녹음은 한 문장이라는 가정; 텍스트/메트릭 추출)
    analysis = analyze_segments(tmp_path, model_name="turbo", language="ko")
    met = {}
    text = analysis.get("text", "").strip()
    if analysis.get("segments"):
        met = analysis["segments"][0].get("metrics", {})

    # 5) 버전 행 INSERT (기본은 is_selected = False)
    version = VoiceSegmentVersion(
        segment_id=seg.id,
        version_no=ver_no,
        text=text or seg.text,  # 텍스트가 비면 기존 문장 유지
        segment_url=seg_url,
        db=_to_float(met.get("dB")),
        pitch_mean_hz=_to_float(met.get("pitch_mean_hz")),
        rate_wpm=_to_float(met.get("rate_wpm")),
        pause_ratio=_to_float(met.get("pause_ratio")),
        prosody_score=_to_float(met.get("prosody_score")),
        feedback=make_feedback(met),
        is_selected=False
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
        "is_selected": version.is_selected,
        "feedback": version.feedback,
        "metrics": {
            "dB": version.db,
            "pitch_mean_hz": version.pitch_mean_hz,
            "rate_wpm": version.rate_wpm,
            "pause_ratio": version.pause_ratio,
            "prosody_score": version.prosody_score,
        }
    }
