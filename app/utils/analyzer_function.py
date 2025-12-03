import numpy as np

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

# ===== 3. 문장 끝 경계 특징 계산 =====
def compute_final_boundary_features_for_segment(
    rms: np.ndarray,
    f0_hz: np.ndarray,
    frame_times: np.ndarray,
    seg_start: float,
    seg_end: float,
    f0_min: float = 1e-3,
):
    """
    하나의 문장(세그먼트)에 대해:
    - baseline 구간(중간 30%)
    - final 구간(마지막 25%, 0.3~0.8초 클램프)
    를 잡고, dB / pitch 특징을 계산합니다.
    """

    # 문장 길이
    seg_length = seg_end - seg_start
    if seg_length <= 0.8:
        return np.nan, np.nan, np.nan , np.nan

    # final 구간 길이 설정 (25% + 0.3~0.8초 클램프)
    W_final = 0.25 * seg_length
    W_final = max(0.3, min(W_final, 0.8))   # 0.3 ~ 0.8초
    W_final = min(W_final, seg_length)         # 전체 길이보다 길지 않게

    final_start = max(seg_start, seg_end - W_final)
    final_end = seg_end
    
    # baseline 구간 (중간 30% 구간)
    base_start = seg_start + 0.3 * seg_length
    base_end = seg_start + 0.6 * seg_length

    # base, final 구간 마스크
    base_masked = (frame_times >= base_start) & (frame_times <= base_end)
    final_masked = (frame_times >= final_start) & (frame_times <= final_end)

    # ---- dB 기준 ----

    # baseline / final 구간 데이터 추출 + 평균 계산
    rms_db_base = rms[base_masked]
    rms_db_final = rms[final_masked]
    rms_db_final_frame_times = frame_times[final_masked]

    mean_db_base = _safe_mean(rms_db_base)
    mean_db_final = _safe_mean(rms_db_final)

    # dB drop 계산
    if np.isnan(mean_db_base) or np.isnan(mean_db_final):
        final_db_drop = np.nan
    else:
        final_db_drop = mean_db_base - mean_db_final  # 양수면 끝이 더 작음

    # dB slope 계산
    final_db_slope = _linear_slope(rms_db_final_frame_times, rms_db_final)


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

        if f0_hz_base_valid.size < 2 or f0_hz_final_valid.size < 2:
            return np.nan, np.nan, np.nan , np.nan

        # 55 Hz 기준 semitone 변환
        f0_base_semitone = 12.0 * np.log2(f0_hz_base_valid / 55.0)
        f0_final_semitone = 12.0 * np.log2(f0_hz_final_valid / 55.0)

        mean_f0_base = _safe_mean(f0_base_semitone)
        mean_f0_final = _safe_mean(f0_final_semitone)

        if np.isnan(mean_f0_base) or np.isnan(mean_f0_final):
            final_pitch_drop = np.nan
        else:
            final_pitch_drop = mean_f0_base - mean_f0_final  # 양수면 내려감

        final_pitch_slope = _linear_slope(f0_hz_final_valid_frame_times, f0_final_semitone)


        final_db_drop = float(final_db_drop) if np.isfinite(final_db_drop) else np.nan,
        final_db_slope = float(final_db_slope) if np.isfinite(final_db_slope) else np.nan,
        final_pitch_drop = float(final_pitch_drop) if np.isfinite(final_pitch_drop) else np.nan,
        final_pitch_slope = float(final_pitch_slope) if np.isfinite(final_pitch_slope) else np.nan,
    
    return final_db_drop[0], final_db_slope[0], final_pitch_drop[0], final_pitch_slope[0]

# ===== 4. CV 기반 레이블링 함수들 =====
## ===== 4-1. Pitch CV =====
def classify_pitch_cv(cv_pitch: float):
    if not np.isfinite(cv_pitch):
        return "UNKNOWN", "유성 구간이 너무 짧거나 피치 측정이 불안정해서 음높이 변동을 평가하기 어렵습니다."

    if cv_pitch < 0.15:
        label = "VERY_LOW"
        comment = (
            "음높이 변동이 매우 적은 편입니다. 전체적으로 단조롭게 들릴 수 있어, "
            "강조가 필요한 부분에서 피치를 조금 더 올리거나 내려 주면 좋습니다."
        )
    elif cv_pitch < 0.23:
        label = "LOW_NORMAL"
        comment = (
            "음높이 변동이 비교적 작은 편입니다. 차분하고 안정적인 인상을 주지만, "
            "설명이나 발표에서는 핵심 키워드에서 피치를 조금 더 변화시키면 전달력이 좋아질 수 있습니다."
        )
    elif cv_pitch < 0.33:
        label = "TYPICAL"
        comment = (
            "일반적인 말하기에서 자주 관찰되는 수준의 음높이 변동입니다. "
            "억양이 자연스러워서 듣기에 무난한 편입니다."
        )
    elif cv_pitch < 0.45:
        label = "HIGH"
        comment = (
            "음높이 변동이 비교적 큰 편입니다. 감정 표현이나 강조가 풍부하게 느껴질 수 있지만, "
            "일부 구간에서는 다소 요동치는 느낌을 줄 수도 있습니다."
        )
    else:
        label = "VERY_HIGH"
        comment = (
            "음높이 변동이 매우 큰 편입니다. 감정이 많이 실린 말하기이거나, "
            "긴장을 많이 한 상태일 수 있습니다. 중요한 부분 위주로만 피치 변화를 크게 사용하면 "
            "전체 흐름이 더 안정적으로 느껴질 수 있습니다."
        )

    return label, comment

## ==== 4-2. Energy CV =====
def classify_energy_cv(cv_energy: float):
    if not np.isfinite(cv_energy):
        return "UNKNOWN", "유성 구간이 너무 짧거나 RMS 측정이 불안정해서 크기 변동을 평가하기 어렵습니다."

    if cv_energy < 0.25:
        label = "VERY_LOW"
        comment = (
            "음성 크기 변동이 매우 적습니다. 전체적으로 볼륨이 거의 일정해서 안정적으로 들리지만, "
            "강조해야 할 단어나 문장에서 크기 변화를 조금 더 주면 전달력이 좋아질 수 있습니다."
        )
    elif cv_energy < 0.45:
        label = "LOW_NORMAL"
        comment = (
            "음성 크기 변동이 비교적 작은 편입니다. 차분한 인상을 주지만, "
            "핵심 부분에서 크기를 살짝 더 키우면 메시지가 더 뚜렷하게 전달될 수 있습니다."
        )
    elif cv_energy < 0.70:
        label = "TYPICAL"
        comment = (
            "일반적인 말하기에서 흔히 보이는 수준의 크기 변동입니다. "
            "문장과 단어에 따라 적당한 강약이 들어가 있어 자연스럽게 들립니다."
        )
    elif cv_energy < 1:
        label = "HIGH"
        comment = (
            "음성 크기 변동이 꽤 큰 편입니다. 강조가 많아서 에너지가 느껴질 수 있지만, "
            "일부 구간에서는 소리가 갑자기 커지거나 작아지는 느낌을 줄 수도 있습니다."
        )
    else:
        label = "VERY_HIGH"
        comment = (
            "음성 크기 변동이 매우 큽니다. 마이크 거리 변화, 몸 움직임, 혹은 볼륨 조절이 "
            "불안정했을 수 있습니다. 중요한 부분을 제외하고는 크기 변화를 조금 줄이면 "
            "청자가 더 편하게 들을 수 있습니다."
        )

    return label, comment

## === 4-3. Rate WPM =====
def classify_rate_wpm(rate_wpm: float):
    if not np.isfinite(rate_wpm):
        return "UNKNOWN", "말하기 속도를 평가하기 어렵습니다."
    
    if rate_wpm < 110:
        label = "SLOW"
        comment = (
            "차분하고 또렷하지만 전반적으로 느린 편. 이해하기 쉬운 속도이지만, 에너지가 낮거나 "
            "흐름이 단조롭게 느껴질 수 있음. 중요한 발표에서는 강조 포인트에서 속도 조절이 필요함."
        )
    elif rate_wpm < 160:
        label = "TYPICAL"
        comment = (
            "발표에서 가장 흔하고 듣기 편한 정상 속도. 내용 이해·집중·전달력의 균형이 좋은 구간."
            "대부분의 프레젠테이션이 이 범위에 들어감."
        )
    else:
        label = "FAST"
        comment = (
            "빠른 말하기. 에너지가 높고 생동감 있지만, 정보량이 많을 경우 청중의 이해가 떨어질 수 있음."
            "긴장으로 인해 빨라졌을 가능성도 있으며, 문장 끝 처리나 뉘앙스가 희생될 수 있음."
        )

    return label, comment

# ===== 5. Ending Pattern =====
# ===== 5-1. labeling =====
def labeling_volume_ending(db_drop: float,
                           db_slope: float):
    if not np.isfinite(db_drop):
        drop_label = "UNKNOWN"
    if db_drop < -4:
        drop_label = "DP_STRONG_RISE"
    elif db_drop < 4:
        drop_label = "DP_WEAK_RISE_OR_FLAT"
    elif db_drop < 8:
        drop_label = "DP_WEAK_FALL"
    else:
        drop_label = "DP_STRONG_FALL"

    if not np.isfinite(db_slope):
        slope_label = "UNKNOWN"
    if db_slope < -1:
        slope_label = "SLOPE_DECAY"
    elif db_slope < 1:
        slope_label = "SLOPE_FLAT"
    else:
        slope_label = "SLOPE_RISE"

    return drop_label, slope_label

def labeling_pitch_ending(pitch_drop: float,
                          pitch_slope: float):
    if not np.isfinite(pitch_drop):
        drop_label = "UNKNOWN"
    if pitch_drop < -3:
        drop_label = "PITCH_STRONG_RISE"
    elif pitch_drop < 1:
        drop_label = "PITCH_RISE_OR_FLAT"
    elif pitch_drop < 3:
        drop_label = "PITCH_NATURAL_FALL"
    else:
        drop_label = "PITCH_STRONG_FALL"

    if not np.isfinite(pitch_slope):
        slope_label = "UNKNOWN"
    if pitch_slope < -1:
        slope_label = "SLOPE_DECAY"
    elif pitch_slope < 1:
        slope_label = "SLOPE_FLAT"
    else:
        slope_label = "SLOPE_RISE"

    return drop_label, slope_label

# ===== 5-2. Classification =====
def classify_volume_ending(final_db_drop: float,
                           final_db_slope: float,):


    db_drop_label, db_slope_label = labeling_volume_ending(
        db_drop=final_db_drop,
        db_slope=final_db_slope
    )

    # 기본값
    final_label = "VOL_END_MIXED"

    # 1) 안정적으로 끝까지 유지
    if (db_drop_label == "DP_WEAK_RISE_OR_FLAT" and
            db_slope_label == "SLOPE_FLAT"):
        final_label = "VOL_END_STABLE_CLEAR"

    # 2) 자연스럽게 살짝 줄어드는 마무리
    elif ((db_drop_label == "DP_WEAK_FALL" and
           db_slope_label in {"SLOPE_FLAT", "SLOPE_DECAY"})
          or
          (db_drop_label == "DP_WEAK_RISE_OR_FLAT" and
           db_slope_label == "SLOPE_DECAY")):
        final_label = "VOL_END_NATURAL_SOFT"

    # 3) 강하게 사라지는 느낌의 마무리
    elif (db_drop_label == "DP_STRONG_FALL" and
          db_slope_label == "SLOPE_DECAY"):
        final_label = "VOL_END_STRONG_FADE"

    # 4) 끝으로 갈수록 볼륨이 올라가는 패턴
    elif (db_drop_label == "DP_STRONG_RISE" or
          db_slope_label == "SLOPE_RISE"):
        final_label = "VOL_END_RISING"

    # 라벨별 코멘트 사전
    comments = {
        "VOL_END_STABLE_CLEAR": (
            "문장 끝까지 볼륨이 거의 유지되어 마지막 단어까지 또렷하게 들리는 패턴입니다. "
            "중요한 내용을 전달할 때는 장점이지만, 모든 문장이 이렇게 끝나면 다소 단조롭게 느껴질 수 있습니다."
        ),
        "VOL_END_NATURAL_SOFT": (
            "문장 끝으로 갈수록 볼륨이 살짝 줄어들어 자연스럽게 마무리되는 패턴입니다. "
            "부담 없이 듣기 좋은 톤으로, 일반적인 진술형 문장 마감에 잘 어울립니다."
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
            "감정 표현이 풍부한 상황에서는 자연스러울 수 있지만, 설명 중심 발표에서는 패턴을 조금 더 일관되게 가져가는 것이 좋습니다."
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

    # 1) 질문형/열린 느낌
    if (pitch_drop_label == "PITCH_STRONG_RISE" or
            pitch_slope_label == "PITCH_SLOPE_RISE"):
        final_label = "PITCH_END_QUESTION_LIKE"

    # 2) 평탄·중립적인 마무리
    elif (pitch_drop_label == "PITCH_RISE_OR_FLAT" and
          pitch_slope_label == "PITCH_SLOPE_FLAT"):
        final_label = "PITCH_END_FLAT_NEUTRAL"

    # 3) 자연스러운 진술형 종결
    elif (pitch_drop_label == "PITCH_NATURAL_FALL" and
          pitch_slope_label in {"PITCH_SLOPE_FLAT", "PITCH_SLOPE_FALL"}):
        final_label = "PITCH_END_NATURAL_DECLARATIVE"

    # 4) 강한 종결감/단호한 마무리
    elif (pitch_drop_label == "PITCH_STRONG_FALL" and
          pitch_slope_label == "PITCH_SLOPE_FALL"):
        final_label = "PITCH_END_STRONG_DECLARATIVE"

    comments = {
        "PITCH_END_QUESTION_LIKE": (
            "문장 끝에서 피치가 올라가거나 유지되면서, 질문하거나 말을 열어두는 느낌을 줍니다. "
            "실제 질문·청유 문장에서는 자연스럽지만, 일반 진술 문장에서 반복되면 말끝이 애매하게 들릴 수 있습니다."
        ),
        "PITCH_END_FLAT_NEUTRAL": (
            "피치 변화가 거의 없이 담백하게 끝나는 패턴입니다. "
            "차분하고 중립적인 인상을 주지만, 발표 전체가 이렇게만 진행되면 감정 표현이 부족하고 단조롭게 느껴질 수 있습니다."
        ),
        "PITCH_END_NATURAL_DECLARATIVE": (
            "피치가 자연스럽게 조금 내려가면서 진술형 문장답게 마무리되는 패턴입니다. "
            "정보 전달과 설명에 적합한 안정적인 종결톤입니다."
        ),
        "PITCH_END_STRONG_DECLARATIVE": (
            "피치가 짧은 구간에서 크게 떨어져 매우 단호하게 문장을 닫는 패턴입니다. "
            "결론이나 핵심 메시지를 강조할 때 효과적이지만, 자주 사용하면 다소 딱딱하거나 강하게 느껴질 수 있습니다."
        ),
        "PITCH_END_MIXED": (
            "문장 끝 피치 패턴이 일관되지 않아, 청자가 종결감이나 질문감을 명확히 인식하기 어렵습니다. "
            "감정 표현이 많은 장면에서는 자연스러울 수 있지만, 설명형 발표에서는 톤 패턴을 조금 더 정돈해 주는 것이 좋습니다."
        )
    }

    return final_label, comments[final_label]
