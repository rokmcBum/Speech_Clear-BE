# LLM 피드백 생성
import json
import os
from app.utils.analyzer_function import classify_energy_cv, classify_pitch_cv, classify_rate_wpm, classify_volume_ending, classify_pitch_ending
from app.domain.llm.service.clova_LLN import get_sentence_feedback_from_LLM

def make_feedback(segments:list):
    # # JSON 파일 로드
    # with open("result1.json", "r", encoding="utf-8") as f:
    #     result = json.load(f)

    index = 0
    results_feedback = []

    # 문장별 분석 결과 추출
    for seg in segments:
        # 초기화
        analyzed = {
            "volume_stability": {},
            "ending_pattern": {},
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
        if label != "NORMAL_VAR":
            need_feedback = True 

        # 3. Rate Level
        rate_level = seg["wpm"]["rate_wpm"]
        label, comment = classify_rate_wpm(rate_level)
        analyzed["rate_level"] = {
            "label": label,
            "comment": comment
        }
        if label != "TYPICAL":
            need_feedback = True

        # 4. Ending Pattern - Energy
        final_rms_ratio = seg["final_boundary"]["final_rms_ratio"]
        final_rms_slope = seg["final_boundary"]["final_rms_slope"]
        vol_end_label, vol_end_comment = classify_volume_ending(final_rms_ratio, final_rms_slope)
        analyzed["ending_pattern"]= {
            "label": vol_end_label,
            "comment": vol_end_comment
        }
        if vol_end_label not in {"VOL_END_NATURAL_SOFT","VOL_END_STABLE_CLEAR"}:
            need_feedback = True

        # 5. Ending Pattern - Pitch
        final_pitch_semitone_drop = seg["final_boundary"]["final_pitch_semitone_drop"]
        final_pitch_semitone_slope = seg["final_boundary"]["final_pitch_semitone_slope"]
        pitch_end_label, pitch_end_comment = classify_pitch_ending(final_pitch_semitone_drop, final_pitch_semitone_slope)
        analyzed["pitch_ending"] = {
            "label": pitch_end_label,
            "comment": pitch_end_comment
        }
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
        results_feedback.append({
            "segment_index": index,
            "start_time": seg["start"],
            "end_time": seg["end"],
            "feedback": feedback,
            "needs_feedback": need_feedback
        })
        index += 1
    
    return results_feedback