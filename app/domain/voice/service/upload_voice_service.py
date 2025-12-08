# app/domain/voice/service.py
import os
import tempfile
import uuid
from typing import Optional

import librosa
import numpy as np
from fastapi import UploadFile
from pydub import AudioSegment
from sqlalchemy.orm import Session

from app.domain.llm.service import make_feedback_service
from app.domain.llm.service.classify_text_service import classify_text_into_sections
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.user.model.user import User
from app.domain.voice.model.voice import Voice, VoiceParagraphFeedback, VoiceSegment
from app.domain.voice.utils.draw_dB_image import draw
from app.domain.voice.utils.map_sections_to_segments import (
    split_llm_sections_into_sentences_with_clova_timestamps,
)
from app.infrastructure.storage.object_storage import upload_file
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
    make_part_index_map,
)
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
    clova_words = clova_result.get("words", [])  # Clova Speech의 word timestamps
    
    if progress_callback:
        progress_callback(30) # Clova Speech 완료: 30%
    
    # 2단계: LLM으로 문단별 분할 (part 정보를 위해)
    llm_sections = []
    try:
        llm_sections = classify_text_into_sections(full_text)
    except Exception as e:
        print(f"⚠️ LLM 분할 실패: {e}")
    
    if progress_callback:
        progress_callback(50) # LLM 완료: 50%

    # 3단계: LLM 문단을 kss로 문장 단위로 분할하고 Clova word timestamps로 시간 계산
    final_segments = []
    try:
        sentence_segments = split_llm_sections_into_sentences_with_clova_timestamps(
            llm_sections=llm_sections,
            clova_words=clova_words
        )
        
        # 문장 단위 segments를 final_segments에 추가
        for seg in sentence_segments:
            final_seg = {
                "start": seg.get("start", 0),  # 밀리초
                "end": seg.get("end", 0),      # 밀리초
                "text": seg.get("text", "").strip(),
                "words": seg.get("words", []),  # [start_ms, end_ms, text] 형태
                "part": seg.get("part")  # part 정보
            }
            final_segments.append(final_seg)
    except Exception as e:
        print(f"⚠️ 문장 분할 실패: {e}, Clova segments를 그대로 사용")
        # 실패 시 Clova segments를 그대로 사용 (fallback)
        clova_segments = clova_result.get("segments", [])
        for seg in clova_segments:
            final_seg = {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "words": seg.get("words", []),
            }
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
        fmax=800,
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

        mean_db = 0.0
        mean_hz = 0.0

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
            
            # [FIX] dB calculation: RMS -> dB
            if len(rms_seg) > 0:
                # ref를 np.max(rms)로 하면 전체 대비 상대 크기
                # 여기서는 rms_seg(해당 세그먼트) 자체의 max를 쓸지, 전체 max를 쓸지 결정 필요
                # 재녹음 로직(analyze_segments)과의 정합성을 위해 세그먼트 기준 계산 시도
                # 하지만, 구간별 비교를 위해 global max가 나을 수 있음
                # 여기서는 오디오 분석 유틸과 가장 유사하게 맞춤
                # analyze_segments에서는 "amplitude_to_db(rms, ref=np.max)"를 씀 (local max)
                mean_db = float(np.mean(librosa.amplitude_to_db(rms_seg, ref=1.0)))

            # pitch 계산
            mean_st, std_st, cv_pitch = compute_pitch_cv_segment(
                f0_hz=f0_seg,
                f0_min=1e-3
            )
            
            # [FIX] Hz calculation
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

        else:
            # y_seg가 없을 경우 기본값
            mean_r, std_r, cv_energy = 0, 0, 0
            mean_st, std_st, cv_pitch = 0, 0, 0
            final_rms_ratio, final_rms_slope = 0, 0
            final_pitch_semitone_drop, final_pitch_semitone_slope = 0, 0
            rate_wpm, words_count = 0, 0

        # 문장 단위 정보 구성
        segment_info ={
            "id": id,
            "part": seg["part"],
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "energy": {
                "mean_rms": round(mean_r, 2),
                "std_rms": round(std_r, 2),
                "cv": round(cv_energy, 4),
                "mean_db": round(mean_db, 2)  # [FIX] Added mean_db
            },
            "pitch": {
                "mean_st": round(mean_st, 2),
                "std_st": round(std_st, 2),
                "cv": round(cv_pitch, 4),
                "mean_hz": round(mean_hz, 2)  # [FIX] Added mean_hz
            },
            "wpm":{
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
            current_progress = 50 + int(id / len(final_segments) * 40)
            progress_callback(current_progress)  
        analyzed.append(segment_info)

    # 문단별 인덱스 맵 생성
    paragraph_index = make_part_index_map(final_segments)

    feedbacks_list, paragraph_feedback = make_feedback_service.make_feedback(analyzed, paragraph_index)

    # 피드백을 segment_index로 매핑 (id는 0부터 시작)
    feedback_map = {fb["id"]: fb["feedback"] for fb in feedbacks_list}
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
    
    # 문단별 피드백 저장
    for para_fb in paragraph_feedback:
        paragraph_feedback_obj = VoiceParagraphFeedback(
            voice_id=voice.id,
            part=para_fb["part"],
            feedback=para_fb["feedback"]
        )
        db.add(paragraph_feedback_obj)
    
    db.flush()  # paragraph_feedback도 flush하여 voice.id 사용 가능하도록
    
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
        
        feedback_text = feedback_map.get(order_no - 1, "")


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
            db=float(energy.get("mean_db", 0.0)),    # [FIX] Use mean_db
            pitch_mean_hz=float(pitch.get("mean_hz", 0.0)), # [FIX] Use mean_hz (now calculated)
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