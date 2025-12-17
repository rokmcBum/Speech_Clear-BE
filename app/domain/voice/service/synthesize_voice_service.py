import os
import tempfile
from typing import Callable, Dict, List, Optional

import librosa
import numpy as np
from fastapi import HTTPException
from pydub import AudioSegment
from sqlalchemy.orm import Session
from starlette import status

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceSegment, VoiceSegmentVersion
from app.domain.voice.utils.voice_permission import verify_voice_ownership
from app.infrastructure.storage.object_storage import download_file, upload_file
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
)


def synthesize_voice(voice_id: int, db: Session, user: User, selections: Optional[List[Dict]] = None, progress_callback: Optional[Callable[[int], None]] = None):
    if progress_callback:
        progress_callback(10)  # 시작: 10%

    # voice 소유권 검증
    original_voice = verify_voice_ownership(voice_id, user, db)
    
    voice_segments = (
        db.query(VoiceSegment)
        .filter(VoiceSegment.voice_id == voice_id)
        .order_by(VoiceSegment.order_no)
        .all()
    )

    if not voice_segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="해당 음성에는 세그먼트가 없습니다."
        )

    # selections를 딕셔너리로 변환 (segment_id -> selected_version_index)
    selections_map = {}
    if selections:
        for sel in selections:
            segment_id = sel.get("segment_id")
            selected_version_index = sel.get("selected_version_index")
            if segment_id is not None and selected_version_index is not None:
                selections_map[segment_id] = selected_version_index

    final_segments = []
    selected_segments_info = []  # 분석을 위한 세그먼트 정보 저장
    for voice_segment in voice_segments:
        segment_id = voice_segment.id
        selected_version_index = selections_map.get(segment_id)
        
        selected_url = None
        selected_text = None
        
        if selected_version_index is None:
            # selections에 없는 경우, 기존 로직대로 최신 버전 사용
            selected_ver = (
                db.query(VoiceSegmentVersion)
                .filter(VoiceSegmentVersion.segment_id == segment_id)
                .order_by(VoiceSegmentVersion.version_no.desc())
                .first()
            )
            if selected_ver:
                selected_url = selected_ver.segment_url
                selected_text = selected_ver.text
            else:
                selected_url = voice_segment.segment_url
                selected_text = voice_segment.text
        elif selected_version_index == -1:
            # 원본 사용
            selected_url = voice_segment.segment_url
            selected_text = voice_segment.text
        else:
            # 재녹음 버전 사용 (selected_version_index: 0=첫 번째, 1=두 번째, ...)
            # version_no는 1부터 시작하므로 selected_version_index + 1 = version_no
            target_version_no = selected_version_index + 1
            selected_ver = (
                db.query(VoiceSegmentVersion)
                .filter(
                    VoiceSegmentVersion.segment_id == segment_id,
                    VoiceSegmentVersion.version_no == target_version_no
                )
                .first()
            )
            if selected_ver:
                selected_url = selected_ver.segment_url
                selected_text = selected_ver.text
            else:
                # 요청한 버전이 없으면 원본 사용
                print(f"[WARN] segment_id={segment_id}의 version_no={target_version_no}이 없습니다. 원본을 사용합니다.")
                selected_url = voice_segment.segment_url
                selected_text = voice_segment.text
        
        final_segments.append(selected_url)
        selected_segments_info.append({
            "url": selected_url,
            "text": selected_text,
            "part": voice_segment.part,
            "order_no": voice_segment.order_no
        })

    if not final_segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="합성할 세그먼트가 없습니다."
        )

    combined = None
    tmp_files = []
    out_tmp = None

    try:
        downloads_count = 0
        total_downloads = len(final_segments)

        for object_name in final_segments:
            # [FIX] NamedTemporaryFile을 close() 하여 핸들 반환 (다른 프로세스/함수에서 접근 가능하도록)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
            tmp.close()
            
            download_file(object_name, tmp.name)
            tmp_files.append(tmp.name)
            
            downloads_count += 1
            if progress_callback:
                 # 다운로드 구간: 10% ~ 30%
                progress = 10 + int((downloads_count / total_downloads) * 20)
                progress_callback(progress)

            try:
                # 1. 포맷 지정 없이 시도 (ffmpeg 자동 감지)
                seg_audio = AudioSegment.from_file(tmp.name)
            except Exception:
                # 2. 실패 시 WebM으로 명시적 시도
                try:
                    seg_audio = AudioSegment.from_file(tmp.name, format="webm")
                except Exception as e:
                    print(f"[WARN] 오디오 디코딩 실패 ({object_name}): {e}")
                    continue

            combined = seg_audio if combined is None else combined + seg_audio
            
        if combined is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="모든 세그먼트 디코딩에 실패했습니다. (파일 손상 또는 다운로드 오류)"
            )
        
        if progress_callback:
            progress_callback(50)  # 병합 완료: 50%

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
        combined.export(out_tmp.name, format="mp4")  # m4a는 mp4 컨테이너로 저장
        out_tmp.close()

        object_name = f"voices/{voice_id}/final/final_{voice_id}.m4a"
        final_url = upload_file(out_tmp.name, object_name)

        if progress_callback:
            progress_callback(70)  # 업로드 완료: 70%

        duration_sec = len(combined) / 1000.0
        
        # 합성된 세그먼트들을 분석하여 total_feedback 생성
        total_feedback = ""
        try:
            analyzed_segments = []
            
            total_analysis = len(selected_segments_info)
            
            for idx, seg_info in enumerate(selected_segments_info):
                seg_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
                try:
                    download_file(seg_info["url"], seg_tmp.name)
                    
                    # STT 수행
                    stt_result = make_voice_to_stt(seg_tmp.name)
                    
                    # 오디오 분석
                    y, sr = librosa.load(seg_tmp.name, sr=16000)
                    frame_length = 2048
                    hop_length = 256
                    
                    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                    f0, _, _ = librosa.pyin(y, fmin=80, fmax=800, frame_length=frame_length, hop_length=hop_length, sr=sr)
                    
                    frame_idx = np.arange(rms.shape[0])
                    frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr
                    
                    # 세그먼트 정보 구성
                    seg_start = stt_result["words"][0]["start"] if stt_result.get("words") else 0.0
                    seg_end = stt_result["words"][-1]["end"] if stt_result.get("words") else len(y) / sr
                    seg_text = stt_result.get("text", seg_info["text"])
                    
                    # words 형식 변환
                    words = []
                    if stt_result.get("words"):
                        for w in stt_result["words"]:
                            words.append([
                                int(w["start"] * 1000),
                                int(w["end"] * 1000),
                                w["text"]
                            ])
                    
                    final_seg = {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                        "words": words
                    }
                    
                    # 유성 마스크 생성
                    full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, [final_seg])
                    
                    # 세그먼트 구간 프레임 인덱스
                    y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)
                    
                    if len(y_seg) > 0:
                        seg_voice_masked = y_seg & full_voice_masked
                        rms_seg = rms[seg_voice_masked]
                        f0_seg = f0[seg_voice_masked]
                        
                        # 에너지 통계
                        mean_r, std_r, cv_energy = compute_energy_stats_segment(rms_seg, silence_thresh=1e-6)
                        mean_db = float(np.mean(librosa.amplitude_to_db(rms_seg, ref=1.0))) if len(rms_seg) > 0 else 0.0
                        
                        # 피치 통계
                        mean_st, std_st, cv_pitch = compute_pitch_cv_segment(f0_hz=f0_seg, f0_min=1e-3)
                        valid_f0 = f0_seg[f0_seg > 1e-3]
                        mean_hz = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
                        
                        # 세그먼트 프레임 시간
                        seg_frame_idx = np.arange(rms[y_seg].shape[0])
                        seg_frame_times = (seg_frame_idx * hop_length + hop_length / 2.0) / sr
                        
                        # 문장 끝 경계 특징
                        final_rms_ratio, final_rms_slope, final_pitch_semitone_drop, final_pitch_semitone_slope = compute_final_boundary_features_for_segment(
                            rms=rms[y_seg],
                            f0_hz=f0[y_seg],
                            voice_masked=full_voice_masked[y_seg],
                            frame_times=seg_frame_times,
                            seg_length=seg_end - seg_start
                        )
                        
                        # 말하기 속도
                        words_count = len(seg_text.split())
                        duration_min = (seg_end - seg_start) / 60
                        rate_wpm = words_count / duration_min if duration_min > 0 else 0
                        
                        segment_info = {
                            "id": idx,
                            "part": seg_info.get("part"),
                            "text": seg_text,
                            "start": seg_start,
                            "end": seg_end,
                            "energy": {
                                "mean_rms": round(mean_r, 2),
                                "std_rms": round(std_r, 2),
                                "cv": round(cv_energy, 4),
                                "mean_db": round(mean_db, 2)
                            },
                            "pitch": {
                                "mean_st": round(mean_st, 2),
                                "std_st": round(std_st, 2),
                                "cv": round(cv_pitch, 4),
                                "mean_hz": round(mean_hz, 2)
                            },
                            "wpm": {
                                "word_count": words_count,
                                "rate_wpm": round(rate_wpm, 1),
                                "duration_sec": round(seg_end - seg_start, 3)
                            },
                            "final_boundary": {
                                "final_rms_ratio": round(final_rms_ratio, 2),
                                "final_rms_slope": round(final_rms_slope, 4),
                                "final_pitch_semitone_drop": round(final_pitch_semitone_drop, 2),
                                "final_pitch_semitone_slope": round(final_pitch_semitone_slope, 4)
                            },
                            "words": []
                        }
                    else:
                        # 기본값
                        segment_info = {
                            "id": idx,
                            "part": seg_info.get("part"),
                            "text": seg_text,
                            "start": seg_start,
                            "end": seg_end,
                            "energy": {"mean_rms": 0, "std_rms": 0, "cv": 0, "mean_db": 0},
                            "pitch": {"mean_st": 0, "std_st": 0, "cv": 0, "mean_hz": 0},
                            "wpm": {"word_count": 0, "rate_wpm": 0, "duration_sec": 0},
                            "final_boundary": {
                                "final_rms_ratio": 0, "final_rms_slope": 0,
                                "final_pitch_semitone_drop": 0, "final_pitch_semitone_slope": 0
                            },
                            "words": []
                        }
                    
                    analyzed_segments.append(segment_info)
                    
                    if progress_callback:
                        # 분석 구간: 70% ~ 90%
                        progress = 70 + int((idx + 1) / total_analysis * 20)
                        progress_callback(progress)

                finally:
                    if os.path.exists(seg_tmp.name):
                        os.remove(seg_tmp.name)
            
            # total_feedback 생성
            if analyzed_segments:
                _, total_feedback = make_feedback_service.make_feedback(analyzed_segments)
        except Exception as e:
            print(f"[WARN] 합성본 total_feedback 생성 실패: {e}")
            total_feedback = ""
        
        synthesized_voice = Voice(
            user_id=user.id,
            category_id=original_voice.category_id,
            name=original_voice.name,  # 원본 voice의 name 사용
            filename=out_tmp.name,
            content_type="audio/mp4",
            original_url=final_url,
            duration_sec=duration_sec,
            previous_voice_id=voice_id,
            total_feedback=total_feedback if total_feedback else ""
        )
        db.add(synthesized_voice)
        db.flush()
        db.commit()

        if progress_callback:
            progress_callback(100)  # 완료: 100%

        return {
            "voice_id": synthesized_voice.id,
            "final_url": final_url,
            "segments_count": len(final_segments),
            "duration_sec": duration_sec,
            "message": "최종 합성본이 성공적으로 생성되었습니다."
        }
    finally:
        # 임시 파일 정리 (예외 발생 시에도 보장)
        for f in tmp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        if out_tmp and os.path.exists(out_tmp.name):
            try:
                os.remove(out_tmp.name)
            except Exception:
                pass

