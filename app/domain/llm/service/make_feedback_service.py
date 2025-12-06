# LLM 피드백 생성
import json
import os
import numpy as np
from app.utils.analyzer_function import (
    classify_energy_cv, 
    classify_pitch_cv, 
    classify_rate_wpm, 
    classify_volume_ending, 
    classify_pitch_ending,
    classify_paragraph_feedback
)
from app.domain.llm.service.clova_LLN import (
    get_sentence_feedback_from_LLM, 
    get_paragraph_feedback_from_LLM
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

def make_feedback(segments:list, paragraph_index:dict):
    # # JSON 파일 로드
    # with open("result1.json", "r", encoding="utf-8") as f:
    #     result = json.load(f)

    index = 0
    sentence_feedback = []
    
    paragraph_analysis = []
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
        matrix_5x5 = update_matrix(matrix_5x5, "volume_pattern", analyzed["volume_pattern"]["label"])
        if vol_end_label not in {"VOL_END_NATURAL_SOFT","VOL_END_STABLE_CLEAR"}:
            need_feedback = True

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
        matrix_5x5 = update_matrix(matrix_5x5, "pitch_ending", analyzed["pitch_ending"]["label"])
        if pitch_end_label not in {"PITCH_END_NATURAL_DECLARATIVE",
                                    "PITCH_END_FLAT_NEUTRAL", 
                                    "PITCH_END_STRONG_DECLARATIVE"}:
            need_feedback = True

        # print(analyzed)
        # LLM 피드백 생성
        if need_feedback:
            # feedback = get_sentence_feedback_from_LLM(analyzed)
            feedback = None
        else:
            feedback = None

        _, end = paragraph_index[seg["part"]]
        # 매트릭스 업데이트
        if index == end:
            paragraph_analysis.append({
                "part": seg["part"],
                "matrix": matrix_5x5.tolist()
            })
            # 매트릭스 초기화
            matrix_5x5 = np.zeros((5, 5), dtype=int)

        sentence_feedback.append({
            "id": index,
            "start_time": seg["start"],
            "end_time": seg["end"],
            "feedback": feedback,
            "needs_feedback": need_feedback
        })
        index += 1
    
    # 문단별 분석 결과 생성
    paragraph_labeling = classify_paragraph_feedback(paragraph_analysis)
    paragraph_feedback = []
    for para in paragraph_labeling:
        feedback = get_paragraph_feedback_from_LLM(para)
        paragraph_feedback.append({
            "part": para["part"],
            "feedback": feedback
        })

    return sentence_feedback, paragraph_feedback