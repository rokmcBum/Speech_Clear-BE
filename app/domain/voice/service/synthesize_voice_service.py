import os
import tempfile
from pydub import AudioSegment
from sqlalchemy.orm import Session

from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment, VoiceSegmentVersion
from app.infrastructure.storage.object_storage import upload_file, download_file


def synthesize_voice(voice_id: int, db: Session, user: User):
    voice_segments = (
        db.query(VoiceSegment)
        .filter(VoiceSegment.voice_id == voice_id)
        .order_by(VoiceSegment.order_no)
        .all()
    )

    if not voice_segments:
        raise ValueError("해당 음성에는 세그먼트가 없습니다.")

    final_segments = []
    for voice_segment in voice_segments:
        selected_ver = (
            db.query(VoiceSegmentVersion)
            .filter(VoiceSegmentVersion.segment_id == voice_segment.id)
            .order_by(VoiceSegmentVersion.version_no.desc())
            .first()
        )
        if selected_ver:
            final_segments.append(selected_ver.segment_url)
        else:
            final_segments.append(voice_segment.segment_url)

    if not final_segments:
        raise ValueError("합성할 세그먼트가 없습니다.")

    combined = None
    tmp_files = []

    for object_name in final_segments:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
        download_file(object_name, tmp.name)
        tmp_files.append(tmp.name)

        try:
            seg_audio = AudioSegment.from_file(tmp.name, format="m4a")
        except Exception as e:
            print(f"[WARN] 오디오 디코딩 실패 ({object_name}): {e}")
            continue

        combined = seg_audio if combined is None else combined + seg_audio

    if combined is None:
        raise ValueError("모든 세그먼트 디코딩에 실패했습니다.")

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
    combined.export(out_tmp.name, format="mp4")  # m4a는 mp4 컨테이너로 저장
    out_tmp.close()

    object_name = f"voices/{voice_id}/final/final_{voice_id}.m4a"
    final_url = upload_file(out_tmp.name, object_name)

    duration_sec= len(combined) / 1000.0
    synthesized_voice = Voice(
        user_id=user.id,
        filename=out_tmp.name,
        content_type="audio/mp4",
        original_url=final_url,
        duration_sec=duration_sec,
        previous_voice_id=voice_id
    )
    db.add(synthesized_voice)
    db.flush()
    db.commit()

    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    os.remove(out_tmp.name)

    return {
        "voice_id": synthesized_voice.id,
        "final_url": final_url,
        "segments_count": len(final_segments),
        "duration_sec": duration_sec,
        "message": "최종 합성본이 성공적으로 생성되었습니다."
    }
