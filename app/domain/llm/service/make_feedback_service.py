# LLM 피드백 생성
import json
import os
import librosa
import numpy as np
from app.utils.analyzer_function import (
    compute_energy_stats_segment,
    compute_final_boundary_features_for_segment,
    compute_pitch_cv_segment,
    get_voiced_mask_from_words,
    classify_energy_cv, 
    classify_pitch_cv, 
    classify_rate_wpm, 
    classify_volume_ending, 
    classify_pitch_ending,
    classify_total_label_from_matrix
)
from app.domain.llm.service.stt_service import make_voice_to_stt
from app.domain.llm.service.clova_LLN import (
    get_sentence_feedback_from_LLM, 
    get_total_feedback_from_LLM,
    get_re_recording_feedback_from_LLM
)

LABEL_SPACES = {
    "volume_stability": ["LOW_VAR", "NORMAL_VAR", "HIGH_VAR", "UNKNOWN"],  # 4개
    "volume_pattern": [
        "VOL_END_STABLE_CLEAR",
        "VOL_END_NATURAL_SOFT",
        "VOL_END_STRONG_FADE",
        "VOL_END_RISING",
        "VOL_END_MIXED",
    ],  # 5개
    "pitch_stability": ["LOW_VAR", "NORMAL_VAR", "HIGH_VAR", "UNKNOWN"],  # 4개
    "pitch_ending": [
        "PITCH_END_QUESTION_LIKE",
        "PITCH_END_FLAT_NEUTRAL",
        "PITCH_END_NATURAL_DECLARATIVE",
        "PITCH_END_STRONG_DECLARATIVE",
        "PITCH_END_MIXED",
    ],  # 5개
    "rate_level": ["SLOW", "TYPICAL", "FAST", "UNKNOWN"],  # 4개
}

# 피드백 label 매트릭스 업데이트 함수
def update_matrix(matrix, feature_name, label_value):
    idx = LABEL_SPACES[feature_name].index(label_value)
    feature_order = list(LABEL_SPACES.keys())  # 순서 고정
    row = feature_order.index(feature_name)
    col = idx
    # matrix 크기 확인
    if row < matrix.shape[0] and col < matrix.shape[1]:
        matrix[row, col] += 1

    return matrix

def make_feedback(segments:list, paragraph_index: dict = None):
    # # JSON 파일 로드
    # with open("result1.json", "r", encoding="utf-8") as f:
    #     result = json.load(f)

    index = 0
    sentence_feedback = []
    
    # n x 5 x 5 카운트 매트릭스 (초기 0)
    matrix_5x5 = np.zeros((5, 5), dtype=int)

    # 문장별 분석 결과 추출
    for seg in segments:
        # 초기화
        analyzed = {
            "volume_stability": {},
            "volume_pattern": {},
            "pitch_stability": {},
            "pitch_ending": {},
            "rate_level": {}
        }
        need_feedback = False

        # CV 값 기반 레이블링
        # 1. Volume Stability
        cv_energy = seg["energy"]["cv"]
        label, comment = classify_energy_cv(cv_energy)
        analyzed["volume_stability"] = {
            "label": label,
            "comment": comment
        }
        matrix_5x5 = update_matrix(matrix_5x5, "volume_stability", label)
        # 피드백 필요한지 여부 체크
        if label != "NORMAL_VAR":
            need_feedback = True

        # 2. Pitch Stability
        cv_pitch = seg["pitch"]["cv"]
        label, comment = classify_pitch_cv(cv_pitch)
        analyzed["pitch_stability"] = {
            "label": label,
            "comment": comment
        }
        matrix_5x5 = update_matrix(matrix_5x5, "pitch_stability", label)
        if label != "NORMAL_VAR":
            need_feedback = True 

        # 3. Rate Level
        rate_level = seg["wpm"]["rate_wpm"]
        label, comment = classify_rate_wpm(rate_level)
        analyzed["rate_level"] = {
            "label": label,
            "comment": comment
        }
        matrix_5x5 = update_matrix(matrix_5x5, "rate_level", label)
        if label != "TYPICAL":
            need_feedback = True

        # 4. Ending Pattern - Energy
        final_rms_ratio = seg["final_boundary"]["final_rms_ratio"]
        final_rms_slope = seg["final_boundary"]["final_rms_slope"]
        if final_rms_ratio == "NaN": # 짧은 문장인 경우
            vol_end_label = "VOL_END_STABLE_CLEAR"
            analyzed["volume_pattern"]= {
                "label": vol_end_label,
                "comment": ""
            }
        else:
            vol_end_label, vol_end_comment = classify_volume_ending(final_rms_ratio, final_rms_slope)
            analyzed["volume_pattern"]= {
                "label": vol_end_label,
                "comment": vol_end_comment
            }
        matrix_5x5 = update_matrix(matrix_5x5, "volume_pattern", analyzed["volume_pattern"]["label"])
        if vol_end_label not in {"VOL_END_NATURAL_SOFT","VOL_END_STABLE_CLEAR"}:
            need_feedback = True

        # 5. Ending Pattern - Pitch
        final_pitch_semitone_drop = seg["final_boundary"]["final_pitch_semitone_drop"]
        final_pitch_semitone_slope = seg["final_boundary"]["final_pitch_semitone_slope"]
        if final_pitch_semitone_drop == "NaN": # 짧은 문장인 경우
            pitch_end_label = "PITCH_END_FLAT_NEUTRAL"
            analyzed["pitch_ending"] = {
                "label": pitch_end_label,
                "comment": ""
            }
        else:
            pitch_end_label, pitch_end_comment = classify_pitch_ending(final_pitch_semitone_drop, final_pitch_semitone_slope)
            analyzed["pitch_ending"] = {
                "label": pitch_end_label,
                "comment": pitch_end_comment
            }
        matrix_5x5 = update_matrix(matrix_5x5, "pitch_ending", analyzed["pitch_ending"]["label"])
        if pitch_end_label not in {"PITCH_END_NATURAL_DECLARATIVE",
                                    "PITCH_END_FLAT_NEUTRAL", 
                                    "PITCH_END_STRONG_DECLARATIVE"}:
            need_feedback = True

        # print(analyzed)
        # LLM 피드백 생성
        if need_feedback:
            feedback = get_sentence_feedback_from_LLM(analyzed)
        else:
            feedback = None

        analyzed_metrics = {
            "volume_stability": analyzed["volume_stability"]["label"],
            "volume_pattern": analyzed["volume_pattern"]["label"],
            "pitch_stability": analyzed["pitch_stability"]["label"],
            "pitch_ending": analyzed["pitch_ending"]["label"],
            "rate_level": analyzed["rate_level"]["label"]
        }

        sentence_feedback.append({
            "id": index,
            "analyzed": analyzed_metrics,
            "start_time": seg["start"],
            "end_time": seg["end"],
            "feedback": feedback,
            "needs_feedback": need_feedback
        })
        index += 1
    
    # 총괄 피드백 생성
    total_labeling = classify_total_label_from_matrix(matrix_5x5)
    total_comment = get_total_feedback_from_LLM(total_labeling)

    # 문단별 피드백 생성
    paragraph_feedback = []
    if paragraph_index:
        for part, (start_idx, end_idx) in paragraph_index.items():
            # 해당 문단에 속한 문장들의 분석 결과 추출
            paragraph_segments = segments[start_idx:end_idx + 1]
            
            # 문단별 매트릭스 생성
            para_matrix = np.zeros((5, 5), dtype=int)
            para_analyzed_list = []
            
            for para_seg in paragraph_segments:
                # 문단 내 문장 분석
                para_analyzed = {
                    "volume_stability": {},
                    "volume_pattern": {},
                    "pitch_stability": {},
                    "pitch_ending": {},
                    "rate_level": {}
                }
                
                # Volume Stability
                cv_energy = para_seg["energy"]["cv"]
                label, _ = classify_energy_cv(cv_energy)
                para_analyzed["volume_stability"]["label"] = label
                para_matrix = update_matrix(para_matrix, "volume_stability", label)
                
                # Pitch Stability
                cv_pitch = para_seg["pitch"]["cv"]
                label, _ = classify_pitch_cv(cv_pitch)
                para_analyzed["pitch_stability"]["label"] = label
                para_matrix = update_matrix(para_matrix, "pitch_stability", label)
                
                # Rate Level
                rate_level = para_seg["wpm"]["rate_wpm"]
                label, _ = classify_rate_wpm(rate_level)
                para_analyzed["rate_level"]["label"] = label
                para_matrix = update_matrix(para_matrix, "rate_level", label)
                
                # Volume Pattern
                final_rms_ratio = para_seg["final_boundary"]["final_rms_ratio"]
                final_rms_slope = para_seg["final_boundary"]["final_rms_slope"]
                if final_rms_ratio == "NaN":
                    para_analyzed["volume_pattern"]["label"] = "VOL_END_STABLE_CLEAR"
                else:
                    vol_end_label, _ = classify_volume_ending(final_rms_ratio, final_rms_slope)
                    para_analyzed["volume_pattern"]["label"] = vol_end_label
                para_matrix = update_matrix(para_matrix, "volume_pattern", para_analyzed["volume_pattern"]["label"])
                
                # Pitch Ending
                final_pitch_semitone_drop = para_seg["final_boundary"]["final_pitch_semitone_drop"]
                final_pitch_semitone_slope = para_seg["final_boundary"]["final_pitch_semitone_slope"]
                if final_pitch_semitone_drop == "NaN":
                    para_analyzed["pitch_ending"]["label"] = "PITCH_END_FLAT_NEUTRAL"
                else:
                    pitch_end_label, _ = classify_pitch_ending(final_pitch_semitone_drop, final_pitch_semitone_slope)
                    para_analyzed["pitch_ending"]["label"] = pitch_end_label
                para_matrix = update_matrix(para_matrix, "pitch_ending", para_analyzed["pitch_ending"]["label"])
                
                para_analyzed_list.append(para_analyzed)
            
            # 문단별 총괄 라벨링
            para_total_labeling = classify_total_label_from_matrix(para_matrix)
            
            # 문단별 LLM 피드백 생성
            para_feedback = get_total_feedback_from_LLM(para_total_labeling)
            
            paragraph_feedback.append({
                "part": part,
                "feedback": para_feedback if para_feedback else ""
            })
    
    return sentence_feedback, paragraph_feedback

def make_re_recording_feedback(sentence_feedback: list, id: int, re_recording_path: str):
    '''
    입력값:
    sentence_feedback: 문장별 피드백 리스트(make_feedback 함수 결과)
    id: 재녹음한 문장의 id
    re_recording_path: 재녹음된 오디오 파일 경로

    반환값: 
    feedback: 재녹음 피드백 문자열
    re_analyzed_metrics: 재녹음된 문장에 대한 분석 결과
    '''
    # STT 변환 수행
    stt_result = make_voice_to_stt(re_recording_path)
    
    ## 문장 단위 세그먼트 생성
    # 시간
    start = int(stt_result["words"][0]["start"] * 1000)  # 밀리초 변환

    # words
    words = []
    for seg in stt_result["words"]:
        words.append([
            int(seg["start"] * 1000),  # 밀리초 변환
            int(seg["end"] * 1000),    # 밀리초 변환
            seg["text"]
        ])
        end = int(seg["end"] * 1000)      # 밀리초 변환

    # 단일 문장 세그먼트
    final_segments = [{
        "start": start,
        "end": end,
        "text": stt_result["text"],
        "words": words
    }]

    #===== 오디오 분석 =====
    # 오디오 로드
    y, sr = librosa.load(re_recording_path, sr=16000)
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

    seg = final_segments[0]

    # 문장 단위 정보 (시간, 텍스트)
    seg_start, seg_end = seg["start"]/1000, seg["end"]/1000
    seg_text = seg["text"].strip()
    result_text += " " + seg_text

    # 이 문장에 속하는 프레임 인덱스
    y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)

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
            "mean_st": round(mean_st, 2),
            "std_st": round(std_st, 2),
            "cv": round(cv_pitch, 4)
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
        }
    }

    analyzed.append(segment_info)

    #===== LLM 피드백 생성 =====
    seg = analyzed[0]
        
    # 초기화
    analyzed = {
        "volume_stability": {},
        "volume_pattern": {},
        "pitch_stability": {},
        "pitch_ending": {},
        "rate_level": {}
    }

    # CV 값 기반 레이블링
    # 1. Volume Stability
    cv_energy = seg["energy"]["cv"]
    label, comment = classify_energy_cv(cv_energy)
    analyzed["volume_stability"] = {
        "label": label,
        "comment": comment
    }

    # 2. Pitch Stability
    cv_pitch = seg["pitch"]["cv"]
    label, comment = classify_pitch_cv(cv_pitch)
    analyzed["pitch_stability"] = {
        "label": label,
        "comment": comment
    }

    # 3. Rate Level
    rate_level = seg["wpm"]["rate_wpm"]
    label, comment = classify_rate_wpm(rate_level)
    analyzed["rate_level"] = {
        "label": label,
        "comment": comment
    }

    # 4. Ending Pattern - Energy
    final_rms_ratio = seg["final_boundary"]["final_rms_ratio"]
    final_rms_slope = seg["final_boundary"]["final_rms_slope"]
    if final_rms_ratio == "NaN": # 짧은 문장인 경우
        analyzed["volume_pattern"]= {
            "label": "VOL_END_STABLE_CLEAR",
            "comment": ""
        }
    else:
        vol_end_label, vol_end_comment = classify_volume_ending(final_rms_ratio, final_rms_slope)
        analyzed["volume_pattern"]= {
            "label": vol_end_label,
            "comment": vol_end_comment
        }

    # 5. Ending Pattern - Pitch
    final_pitch_semitone_drop = seg["final_boundary"]["final_pitch_semitone_drop"]
    final_pitch_semitone_slope = seg["final_boundary"]["final_pitch_semitone_slope"]
    if final_pitch_semitone_drop == "NaN": # 짧은 문장인 경우
        analyzed["pitch_ending"] = {
            "label": "PITCH_END_FLAT_NEUTRAL",
            "comment": ""
        }
    else:
        pitch_end_label, pitch_end_comment = classify_pitch_ending(final_pitch_semitone_drop, final_pitch_semitone_slope)
        analyzed["pitch_ending"] = {
            "label": pitch_end_label,
            "comment": pitch_end_comment
        }


    re_analyzed_metrics = {
        "volume_stability": analyzed["volume_stability"]["label"],
        "volume_pattern": analyzed["volume_pattern"]["label"],
        "pitch_stability": analyzed["pitch_stability"]["label"],
        "pitch_ending": analyzed["pitch_ending"]["label"],
        "rate_level": analyzed["rate_level"]["label"]
    }  ## 디버깅용

    # 원본 세그먼트의 analyzed_metrics 찾기
    analyzed_metrics = None
    print(f"[DEBUG] make_re_recording_feedback: 찾는 id={id}, sentence_feedback 개수={len(sentence_feedback)}")
    print(f"[DEBUG] sentence_feedback ids: {[s['id'] for s in sentence_feedback]}")
    
    for seg in sentence_feedback:
        if seg["id"] == id:
            analyzed_metrics = seg["analyzed"]
            print(f"[DEBUG] analyzed_metrics 찾음: {analyzed_metrics}")
            break
    
    # analyzed_metrics를 찾지 못한 경우 에러 처리
    if analyzed_metrics is None:
        print(f"[WARN] sentence_feedback에서 id={id}를 찾을 수 없습니다.")
        print(f"[DEBUG] sentence_feedback 전체: {sentence_feedback}")
        # 기본값 설정
        analyzed_metrics = {
            "volume_stability": "UNKNOWN",
            "volume_pattern": "VOL_END_MIXED",
            "pitch_stability": "UNKNOWN",
            "pitch_ending": "PITCH_END_MIXED",
            "rate_level": "UNKNOWN"
        }

    print(f"[DEBUG] LLM 호출 전 - original: {analyzed_metrics}, new: {re_analyzed_metrics}")
    
    # LLM 피드백 생성
    feedback = get_re_recording_feedback_from_LLM(analyzed_metrics, re_analyzed_metrics)
    
    print(f"[DEBUG] LLM 응답 받음: feedback type={type(feedback)}, length={len(feedback) if feedback else 0}")
    print(f"[DEBUG] LLM 응답 내용 (처음 200자): {str(feedback)[:200] if feedback else 'None'}")
    
    # feedback이 None이거나 빈 문자열인 경우 처리
    if feedback is None or (isinstance(feedback, str) and feedback.strip() == ""):
        print(f"[WARN] LLM 피드백이 비어있습니다. feedback={feedback}")
        feedback = "재녹음 분석이 완료되었습니다."

    return feedback, re_analyzed_metrics

