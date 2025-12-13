from typing import Dict, Any
from numpy.typing import NDArray
import numpy as np

# ===== 공통 유틸 함수들 =====
def make_part_index_map(segments: list):
    part_ranges = {}
    current_part = None
    start_idx = 0

    for idx, seg in enumerate(segments):
        part = seg.get("part")

        if current_part is None:
            # 첫 파트 시작
            current_part = part
            start_idx = idx
        elif part != current_part:
            # 파트가 바뀌면 구간 저장
            part_ranges[current_part] = [start_idx, idx - 1]
            # 새 파트 시작
            current_part = part
            start_idx = idx

    # 마지막 파트도 저장
    if current_part is not None:
        part_ranges[current_part] = [start_idx, len(segments) - 1]

    return part_ranges

# ===== 공통: 유성(발화) 마스크 생성 =====
def get_voiced_mask_from_words(rms: np.ndarray,
                               sr: int,
                               hop_length: int,
                               stt_json: dict,
                               margin: float = 0.03) -> np.ndarray:
    """
    STT JSON에 들어있는 word 단위 타임라인을 이용해서
    '유성(말한) 구간'만 True인 프레임 마스크를 만든다.

    - rms        : librosa.feature.rms(y=...) 결과 (shape: (1, T) 또는 (T,))
    - sr         : 샘플링 레이트
    - hop_length : rms 계산에 사용한 hop_length
    - stt_json   : 네가 준 형태의 STT 결과(dict)
                   stt_json["segments"][i]["words"][j]["start"|"end"]
    - margin     : 각 단어 앞뒤로 포함할 여유 시간(초)

    반환값:
    - mask: shape (T,) 의 bool 배열
    """
    # 정규화 (확인용)
    r = np.asarray(rms).astype(float).squeeze()
    num_frames = r.shape[0]

    # 각 프레임 중심 시간 (초 단위)
    frame_idx = np.arange(num_frames)
    frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr

    mask = np.zeros(num_frames, dtype=bool)

    # 모든 segment/word의 (start, end) 구간을 모아서 mask에 반영
    for seg in stt_json:
        for w in seg["words"]:
            start = w[0]/1000  # ms -> s
            end = w[1]/1000 # ms -> s

            s = max(0.0, start - margin)
            e = end + margin

            mask |= (frame_times >= s) & (frame_times <= e)

    return mask

# ===== 1. Energy CV (유성 구간만 사용, RMS linear) =====
def compute_energy_stats_segment(rms: np.ndarray,
                                 silence_thresh: float = 1e-6):
    """
    rms        : 전체 음성에 대해 계산된 RMS (shape: (T,))
    seg_mask   : 이 segment(문장)에 해당하는 프레임만 True인 bool mask
                 (예: in_seg & voiced_mask_global 같은 것)
    silence_thresh : 실질적 무음으로 간주할 RMS threshold

    반환:
      mean_r, std_r, cv_energy
      (유효 프레임 < 2이면 모두 np.nan)
    """
    # 실질적인 무음 제거 + NaN 제거
    valid = np.isfinite(rms) & (rms > silence_thresh)
    r = rms[valid]

    if r.size < 2:
        return np.nan, np.nan, np.nan

    mean_r = float(np.mean(r))
    std_r = float(np.std(r))

    if mean_r <= 0:
        return mean_r, std_r, np.nan

    cv_energy = std_r / mean_r
    return mean_r, std_r, float(cv_energy)

# ===== 2. Pitch CV (유성 구간만 사용) =====
def compute_pitch_cv_segment(f0_hz: np.ndarray,
                             f0_min: float = 1e-3):
    """
    f0_hz   : 전체 음성에 대해 pyin 등으로 얻은 F0(Hz), NaN 포함 가능 (shape: (T,))
    seg_mask: 이 segment(문장)에 해당하는 프레임만 True인 bool mask
              (예: in_seg & voiced_mask_global 같이, 이미 유성/무성까지 반영된 mask)
    f0_min  : 0 또는 비정상적으로 작은 F0 필터링용 threshold

    반환:
      mean_st, std_st, cv_pitch  (semitone 스케일 기준)
      (유효 프레임 < 2이면 모두 np.nan)
    """
    # NaN 제거 + 0 / 비정상적으로 작은 값 제거
    valid = np.isfinite(f0_hz) & (f0_hz > f0_min)
    f0 = f0_hz[valid]

    if f0.size < 2:
        return np.nan, np.nan, np.nan

    # 55 Hz 기준 semitone 변환
    f0_semitone = 12.0 * np.log2(f0 / 55.0)
    mean_st = float(np.mean(f0_semitone))
    std_st = float(np.std(f0_semitone))

    if mean_st <= 0:
        return mean_st, std_st, np.nan

    cv_pitch = std_st / mean_st
    return mean_st, std_st, float(cv_pitch)

# 평균 계산 (NaN 무시)
def _safe_mean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))

# 기울기 계산 함수
def _linear_slope(x, y):
    """x: 시간(초), y: 값(dB 또는 semitone)
       x,y 길이가 너무 짧으면 NaN 반환
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan
    # 0 기준으로 이동하면 수치적으로 조금 안정적
    x0 = x - x[0]
    coef = np.polyfit(x0, y, 1)
    return float(coef[0])  # 기울기 (단위: y / sec)

# 1D 이동평균 스무딩
def _smooth_1d(x, window=5):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    window = max(1, min(window, x.size))
    if window == 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

# ===== 3. 문장 끝 경계 특징 계산 =====
def compute_final_boundary_features_for_segment(
    rms: np.ndarray,
    f0_hz: np.ndarray,
    voice_masked: np.ndarray, # 유성 구간만 True
    frame_times: np.ndarray,
    seg_length: float,
    f0_min: float = 1e-3,
    smooth_win: int = 5
):
    """
    하나의 문장(세그먼트)에 대해:
    - baseline 구간(중간 30%)
    - final 구간(마지막 25%, 0.3~0.8초 클램프)
    를 잡고, dB / pitch 특징을 계산합니다.
    """

    # 문장 길이
    if seg_length <= 0.8:
        return np.nan, np.nan, np.nan , np.nan

    # final 구간 길이 설정 (25% + 0.3~0.8초 클램프)
    W_final = 0.25 * seg_length
    W_final = max(0.3, min(W_final, 0.8))   # 0.3 ~ 0.8초
    W_final = min(W_final, seg_length)         # 전체 길이보다 길지 않게

    final_start = max(0, seg_length - W_final)
    final_end = seg_length
    
    # baseline 구간 (중간 30% 구간)
    base_start = 0.3 * seg_length
    base_end = 0.6 * seg_length

    # base, final 구간 마스크
    base_masked = (frame_times >= base_start) & (frame_times <= base_end)
    final_masked = (frame_times >= final_start) & (frame_times <= final_end)

    # 유성 구간만 필터링
    base_masked = base_masked & voice_masked
    final_masked = final_masked & voice_masked

    # ---- dB 기준 ----

    # baseline / final 구간 데이터 추출 + 평균 계산
    rms_base = rms[base_masked]
    rms_final = rms[final_masked]
    rms_final_frame_times = frame_times[final_masked]

    # mean_rms_base = _safe_mean(rms_base)
    # mean_rms_final = _safe_mean(rms_final)
    mean_rms_base = float(np.median(rms_base))
    mean_rms_final = float(np.median(rms_final))

    # dB drop 계산
    if np.isnan(mean_rms_base) or np.isnan(mean_rms_final) or mean_rms_base <= 0:
        final_rms_drop = np.nan
        final_rms_ratio = np.nan
    else:
        final_rms_drop = mean_rms_base - mean_rms_final  # 양수면 끝이 더 작음
        final_rms_ratio = mean_rms_final / mean_rms_base

        # dB slope 계산용 스무딩
        if rms_final.size >= 3:
            rms_final_sm = _smooth_1d(rms_final, window=min(smooth_win, rms_final.size))
            final_rms_slope = _linear_slope(rms_final_frame_times, rms_final_sm)
        else:
            final_rms_slope = np.nan


    # ---- Pitch 기준 (semitone) ----
    if f0_hz.size == 0:
        final_pitch_drop = np.nan
        final_pitch_slope = np.nan
    else:
        # baseline / final 구간 데이터 추출 + 유효값 필터링
        f0_hz_base = f0_hz[base_masked]
        f0_hz_final = f0_hz[final_masked]
        f0_hz_final_frame_times = frame_times[final_masked]

        valid = np.isfinite(f0_hz_base) & (f0_hz_base > f0_min)
        f0_hz_base_valid = f0_hz_base[valid]

        valid = np.isfinite(f0_hz_final) & (f0_hz_final > f0_min)
        f0_hz_final_valid = f0_hz_final[valid]
        f0_hz_final_valid_frame_times = f0_hz_final_frame_times[valid]

        if f0_hz_base_valid.size < 3 or f0_hz_final_valid.size < 3:
            final_pitch_drop = np.nan
            final_pitch_slope = np.nan
        else:
            # 55 Hz 기준 semitone 변환
            st_base = 12.0 * np.log2(f0_hz_base_valid / 55.0)
            st_final = 12.0 * np.log2(f0_hz_final_valid / 55.0)

            # drop은 mean 대신 median 사용 → outlier에 덜 민감
            med_f0_base  = float(np.median(st_base))
            med_f0_final = float(np.median(st_final))
            final_pitch_drop = med_f0_base - med_f0_final

            # final 구간 smoothing (값 튐 방지)
            if f0_hz_final_valid_frame_times.size >= 3:
                st_final_sm = _smooth_1d(st_final, window=min(smooth_win, st_final.size))
                final_pitch_slope = _linear_slope(f0_hz_final_valid_frame_times, st_final_sm)
            else:
                final_pitch_slope = np.nan
    
    return final_rms_ratio, final_rms_slope, final_pitch_drop, final_pitch_slope

# ===== 4. CV 기반 레이블링 함수들 =====
## ===== 4-1. Pitch CV =====
def classify_pitch_cv(cv_pitch: float):
    if not np.isfinite(cv_pitch):
        return "UNKNOWN", "유성 구간이 너무 짧거나 피치 측정이 불안정해서 음높이 변동을 평가하기 어렵습니다."

    if cv_pitch < 0.15:
        label = "LOW_VAR"
        comment = (
            "음높이 변동이 상당히 적은 편입니다. 전체적으로 단조롭게 들릴 수 있어서, "
            "강조해야 하는 단어나 문장에서는 피치를 조금 더 올리거나 내려 주면 전달력이 좋아질 수 있습니다."
        )
    elif cv_pitch <= 0.50:
        label = "NORMAL_VAR"
        comment = ""
    else:
        label = "HIGH_VAR"
        comment = (
            "음높이 변동이 꽤 큰 편입니다. 감정 표현이나 에너지가 잘 드러난다는 장점이 있지만, "
            "정보 전달이 중요한 발표에서는 다소 요동치는 느낌을 줄 수 있습니다. "
            "핵심이 아닌 부분에서는 피치 변화를 조금 줄이고, 중요한 문장에만 억양을 집중시키면 더 안정적으로 들릴 수 있습니다."
        )

    return label, comment

## ==== 4-2. Energy CV =====
def classify_energy_cv(cv_energy: float):
    if not np.isfinite(cv_energy):
        return "UNKNOWN", "유성 구간이 너무 짧거나 RMS 측정이 불안정해서 크기 변동을 평가하기 어렵습니다."

    if cv_energy < 0.25:
        label = "LOW_VAR"
        comment = (
            "음성 크기 변동이 상당히 작은 편입니다. 전체적으로 안정적이지만, "
            "강조해야 할 단어나 문장에서 볼륨 차이가 부족해 다소 단조롭게 들릴 수 있습니다. "
            "핵심 키워드에서만 살짝 크기를 올려 주면 더 전달력이 좋아질 수 있습니다."
        )
    elif cv_energy <= 1.00:
        label = "NORMAL_VAR"
        comment = ""
    else:
        label = "HIGH_VAR"
        comment = (
            "음성 크기 변동이 꽤 큰 편입니다. 문장 안에서 볼륨이 많이 출렁여서, "
            "마이크 거리 변화나 몸 움직임, 혹은 긴장으로 인한 볼륨 조절 문제일 가능성이 있습니다. "
            "중요한 부분이 아닌 곳에서는 크기 변화를 조금 줄이고, 문장 전체를 보다 일정한 에너지로 유지하면 "
            "청자가 더 편안하게 들을 수 있습니다."
        )

    return label, comment

## === 4-3. Rate WPM =====
def classify_rate_wpm(rate_wpm: float):
    if not np.isfinite(rate_wpm):
        return "UNKNOWN", "말하기 속도를 평가하기 어렵습니다."
    
    if rate_wpm < 100:
        label = "SLOW"
        comment = (
            "차분하고 또렷하지만 전반적으로 느린 편. 이해하기 쉬운 속도이지만, 에너지가 낮거나 "
            "흐름이 단조롭게 느껴질 수 있음. 중요한 발표에서는 강조 포인트에서 속도 조절이 필요함."
        )
    elif rate_wpm < 190:
        label = "TYPICAL"
        comment = ""
    else:
        label = "FAST"
        comment = (
            "빠른 말하기. 에너지가 높고 생동감 있지만, 정보량이 많을 경우 청중의 이해가 떨어질 수 있음."
            "긴장으로 인해 빨라졌을 가능성도 있으며, 문장 끝 처리나 뉘앙스가 희생될 수 있음."
        )

    return label, comment

# ===== 5. Ending Pattern =====
# ===== 5-1. labeling =====
def labeling_volume_ending(rms_ratio: float,
                           rms_slope: float):
    """
    rms_ratio : final_mean / base_mean (linear RMS)
    rms_slope : final 구간 RMS 선형값 기울기 (amp/sec) - 현재는 참고용
    """
    # --- drop 라벨: 비율만 사용해서 매우 단순화 ---
    if not np.isfinite(rms_ratio):
        drop_label = "DP_UNKNOWN"
    elif rms_ratio <= 0.5:
        # 약 -6 dB 이상 떨어짐 -> 강한 페이드 아웃
        drop_label = "DP_STRONG_FALL"
    elif rms_ratio <= 0.8:
        # 약 -6 ~ -2 dB 사이 -> 자연스러운 소프트 페이드
        drop_label = "DP_SOFT_FALL"
    elif rms_ratio < 1.25:
        # 약 -2 ~ +2 dB 안쪽 -> 거의 유지
        drop_label = "DP_STABLE"
    else:
        # +2 dB 이상 -> 말끝이 분명히 커짐
        drop_label = "DP_RISE"

    # --- slope 라벨: 지금은 세밀하게 쓰지 않고, 기본 정보만 남김 ---
    if not np.isfinite(rms_slope):
        slope_label = "SLOPE_UNKNOWN"
    elif rms_slope <= -0.03:
        slope_label = "SLOPE_DECAY"
    elif rms_slope >= 0.03:
        slope_label = "SLOPE_RISE"
    else:
        slope_label = "SLOPE_FLAT"

    return drop_label, slope_label

def labeling_pitch_ending(pitch_drop: float,
                          pitch_slope: float):
    """
    pitch_drop : baseline mean - final mean (semitone)
                 양수면 말끝에서 Pitch가 내려간 것
    pitch_slope: final 구간에서의 기울기 (st/sec) - 참고용
    """

    # drop 라벨 (drop 중심, slope는 크게 안 씀)
    if not np.isfinite(pitch_drop):
        drop_label = "PITCH_DROP_UNKNOWN"
    elif pitch_drop <= -1.5:
        # final이 baseline보다 1.5 st 이상 높음 -> 질문/열린 느낌
        drop_label = "PITCH_STRONG_RISE"
    elif pitch_drop < 1.5:
        # +/- 1.5 st 안쪽 -> 거의 평탄 or 약간의 흔들림
        drop_label = "PITCH_RISE_OR_FLAT"
    elif pitch_drop < 4.5:
        # 1.5 ~ 4.5 st 하강 -> 자연스러운 진술형 하강
        drop_label = "PITCH_NATURAL_FALL"
    else:
        # 4.5 st 이상 하강 -> 강한 종결감
        drop_label = "PITCH_STRONG_FALL"

    # slope 라벨 (강한 질문/강한 하강 보정용 정도로만 사용 가능)
    if not np.isfinite(pitch_slope):
        slope_label = "PITCH_SLOPE_UNKNOWN"
    elif pitch_slope <= -3.0:
        slope_label = "PITCH_SLOPE_FALL"
    elif pitch_slope >= 3.0:
        slope_label = "PITCH_SLOPE_RISE"
    else:
        slope_label = "PITCH_SLOPE_FLAT"

    return drop_label, slope_label

# ===== 5-2. Classification =====
def classify_volume_ending(final_db_drop: float,
                           final_db_slope: float,):


    db_drop_label, db_slope_label = labeling_volume_ending(
        rms_ratio=final_db_drop,
        rms_slope=final_db_slope
    )

    # 기본값
    final_label = "VOL_END_MIXED"

    if db_drop_label == "DP_STRONG_FALL":
        final_label = "VOL_END_STRONG_FADE"
    elif db_drop_label == "DP_SOFT_FALL":
        final_label = "VOL_END_NATURAL_SOFT"
    elif db_drop_label == "DP_STABLE":
        final_label = "VOL_END_STABLE_CLEAR"
    elif db_drop_label == "DP_RISE":
        final_label = "VOL_END_RISING"

    comments = {
        "VOL_END_STABLE_CLEAR": (""
            # "문장 끝까지 볼륨이 거의 유지되어 마지막 단어까지 또렷하게 들리는 패턴입니다. "
            # "중요한 내용을 강조할 때는 장점이지만, 모든 문장이 이렇게 끝나면 다소 단조롭게 느껴질 수 있습니다."
        ),
        "VOL_END_NATURAL_SOFT": (""
            # "문장 끝으로 갈수록 볼륨이 살짝 줄어들어 자연스럽게 마무리되는 패턴입니다. "
            # "부담 없이 듣기 좋은 톤으로, 일반적인 진술형 문장 마감에 잘 어울립니다."
        ),
        "VOL_END_STRONG_FADE": (
            "문장 끝에서 볼륨이 크게 떨어져 마지막 단어가 흐릿하게 들릴 수 있는 패턴입니다. "
            "긴장하거나 호흡이 부족할 때 자주 나타나며, 중요한 키워드는 끝까지 조금 더 힘을 유지하는 연습이 필요합니다."
        ),
        "VOL_END_RISING": (
            "문장 끝으로 갈수록 볼륨이 올라가는 패턴입니다. 질문하거나 강조할 때는 효과적일 수 있지만, "
            "일반적인 설명 문장에서 반복되면 다소 공격적이거나 어색하게 느껴질 수 있습니다."
        ),
        "VOL_END_MIXED": (
            "문장 끝에서 볼륨 패턴이 일정하지 않아, 청자가 마무리의 인상을 명확하게 느끼기 어렵습니다. "
            "감정 표현이 많은 상황에서는 자연스러울 수 있지만, 설명 중심 발표에서는 패턴을 조금 더 일관되게 "
            "가져가는 것이 좋습니다."
        )
    }

    return final_label, comments[final_label]

def classify_pitch_ending(final_pitch_drop: str,
                          final_pitch_slope: str):
    
    pitch_drop_label, pitch_slope_label = labeling_pitch_ending(
        pitch_drop=final_pitch_drop,
        pitch_slope=final_pitch_slope
    )

    # 기본값
    final_label = "PITCH_END_MIXED"

    # 1) 질문형/열린 느낌 (끝에서 확실히 올라가거나 유지)
    if pitch_drop_label == "PITCH_STRONG_RISE":
        final_label = "PITCH_END_QUESTION_LIKE"

    # 2) 강한 종결감 / 단호한 마무리
    elif pitch_drop_label == "PITCH_STRONG_FALL":
        final_label = "PITCH_END_STRONG_DECLARATIVE"

    # 3) 자연스러운 진술형 종결
    elif pitch_drop_label == "PITCH_NATURAL_FALL":
        final_label = "PITCH_END_NATURAL_DECLARATIVE"

    # 4) 평탄·중립적인 마무리
    elif pitch_drop_label == "PITCH_RISE_OR_FLAT":
        final_label = "PITCH_END_FLAT_NEUTRAL"

    comments = {
        "PITCH_END_QUESTION_LIKE": (
            "문장 끝에서 피치가 올라가거나 유지되면서, 질문하거나 말을 열어두는 느낌을 줍니다. "
            "실제 질문·청유 문장에서는 자연스럽지만, 일반 진술 문장에서 반복되면 말끝이 애매하게 들릴 수 있습니다."
        ),
        "PITCH_END_FLAT_NEUTRAL": (""
            # "피치 변화가 거의 없이 담백하게 끝나는 패턴입니다. "
            # "차분하고 중립적인 인상을 주지만, 발표 전체가 이렇게만 진행되면 감정 표현이 부족하고 "
            # "단조롭게 느껴질 수 있습니다."
        ),
        "PITCH_END_NATURAL_DECLARATIVE": (""
            # "피치가 자연스럽게 조금 내려가면서 진술형 문장답게 마무리되는 패턴입니다. "
            # "정보 전달과 설명에 적합한 안정적인 종결톤입니다."
        ),
        "PITCH_END_STRONG_DECLARATIVE": (""
            # "피치가 짧은 구간에서 크게 떨어져 매우 단호하게 문장을 닫는 패턴입니다. "
            # "결론이나 핵심 메시지를 강조할 때 효과적이지만, 자주 사용하면 다소 딱딱하거나 강하게 느껴질 수 있습니다."
        ),
        "PITCH_END_MIXED": (
            "문장 끝 피치 패턴이 일관되지 않아, 청자가 종결감이나 질문감을 명확히 인식하기 어렵습니다. "
            "감정 표현이 많은 장면에서는 자연스러울 수 있지만, 설명형 발표에서는 톤 패턴을 조금 더 정돈해 주는 것이 좋습니다."
        )
    }

    return final_label, comments[final_label]

# ===== 6. 문장별 라벨을 이용한 문단 피드백 생성 =====
def _ratio(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0

def classify_total_label_from_matrix(mat: NDArray[Any]) -> Dict[str, str]:
    """
    mat: shape (5, 5)의 matrix_5x5

    row 0: volume_stability  [LOW_VAR, NORMAL_VAR, HIGH_VAR, UNKNOWN, _]
    row 1: volume_pattern    [STABLE_CLEAR, NATURAL_SOFT, STRONG_FADE, RISING, MIXED]
    row 2: pitch_stability   [LOW_VAR, NORMAL_VAR, HIGH_VAR, UNKNOWN, _]
    row 3: pitch_ending      [QUESTION_LIKE, FLAT_NEUTRAL, NATURAL_DECL, STRONG_DECL, MIXED]
    row 4: rate_level        [SLOW, TYPICAL, FAST, UNKNOWN, _]

    반환:
    {
        "volume_stability": str,  # "stable" / "moderate" / "unstable"
        "ending_pattern": str,    # "mostly_weak" / "mostly_clear" / "mixed"
        "pitch_stability": str,   # "flat" / "balanced" / "very_dynamic"
        "pitch_ending": str,      # "mostly_falling" / "mostly_flat" / "many_rising" / "mixed"
        "rate_level": str         # "mostly_slow" / "mostly_normal" / "mostly_fast"
    }
    """

    mat = np.asarray(mat)
    if mat.shape[0] < 5 or mat.shape[1] < 5:
        raise ValueError("matrix_5x5는 최소 (5, 5) shape이어야 합니다.")

    # 1) volume_stability → stable / moderate / unstable
    vs = mat[0]
    vs_low, vs_norm, vs_high = int(vs[0]), int(vs[1]), int(vs[2])
    vs_total = vs_low + vs_norm + vs_high

    r_low  = _ratio(vs_low, vs_total)
    r_norm = _ratio(vs_norm, vs_total)
    r_high = _ratio(vs_high, vs_total)

    if r_high >= 0.4:
        volume_stability = "unstable"
    elif r_norm >= 0.4:
        volume_stability = "moderate"
    else:
        # 대부분 LOW_VAR이거나 섞여 있지만 HIGH_VAR는 적을 때
        volume_stability = "stable"

    # 2) volume_pattern → mostly_weak / mostly_clear / mixed
    vp = mat[1]
    vp_stable_clear  = int(vp[0])
    vp_natural_soft  = int(vp[1])
    vp_strong_fade   = int(vp[2])
    vp_rising        = int(vp[3])
    vp_mixed         = int(vp[4])

    weak_count  = vp_strong_fade                   # 힘 확 빠지는 종결
    clear_count = vp_stable_clear + vp_natural_soft
    vp_total    = weak_count + clear_count + vp_rising + vp_mixed

    r_weak  = _ratio(weak_count, vp_total)
    r_clear = _ratio(clear_count, vp_total)

    if r_weak >= 0.5:
        volume_pattern = "mostly_weak"
    elif r_clear >= 0.5:
        volume_pattern = "mostly_clear"
    else:
        volume_pattern = "mixed"

    # 3) pitch_stability → flat / balanced / very_dynamic
    ps = mat[2]
    ps_low, ps_norm, ps_high = int(ps[0]), int(ps[1]), int(ps[2])
    ps_total = ps_low + ps_norm + ps_high

    r_p_low  = _ratio(ps_low, ps_total)
    r_p_norm = _ratio(ps_norm, ps_total)
    r_p_high = _ratio(ps_high, ps_total)

    if r_p_high >= 0.4:
        pitch_stability = "very_dynamic"
    elif r_p_low >= 0.4:
        pitch_stability = "flat"
    else:
        pitch_stability = "balanced"

    # 4) pitch_ending → mostly_falling / mostly_flat / many_rising / mixed
    pe = mat[3]
    pe_qlike  = int(pe[0])  # QUESTION_LIKE → rising
    pe_flat   = int(pe[1])  # FLAT_NEUTRAL
    pe_nat    = int(pe[2])  # NATURAL_DECL
    pe_strong = int(pe[3])  # STRONG_DECL
    pe_mixed  = int(pe[4])

    rising_cnt  = pe_qlike
    falling_cnt = pe_nat + pe_strong
    flat_cnt    = pe_flat

    pe_total = rising_cnt + falling_cnt + flat_cnt + pe_mixed

    r_rising  = _ratio(rising_cnt, pe_total)
    r_falling = _ratio(falling_cnt, pe_total)
    r_flat    = _ratio(flat_cnt, pe_total)

    if r_rising >= 0.5:
        pitch_ending = "many_rising"
    elif r_falling >= 0.5:
        pitch_ending = "mostly_falling"
    elif r_flat >= 0.5:
        pitch_ending = "mostly_flat"
    else:
        pitch_ending = "mixed"

    # 5) rate_level → mostly_slow / mostly_normal / mostly_fast
    rl = mat[4]
    slow, typical, fast = int(rl[0]), int(rl[1]), int(rl[2])
    rl_total = slow + typical + fast

    r_slow    = _ratio(slow, rl_total)
    r_typical = _ratio(typical, rl_total)
    r_fast    = _ratio(fast, rl_total)

    if r_fast >= 0.5:
        rate_level = "mostly_fast"
    elif r_slow >= 0.5:
        rate_level = "mostly_slow"
    else:
        # typical 비율이 가장 크거나,
        # slow/fast가 섞여 있어도 보통은 "정상 범위"로 본다
        rate_level = "mostly_normal"

    return {
        "volume_stability": volume_stability,
        "ending_pattern":   volume_pattern,
        "pitch_stability":  pitch_stability,
        "pitch_ending":     pitch_ending,
        "rate_level":       rate_level,
    }
