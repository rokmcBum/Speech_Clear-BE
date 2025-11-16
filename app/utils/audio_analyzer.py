# app/utils/audio_analyzer.py
import whisper
import librosa
import numpy as np


def transcribe_audio(audio_path: str, model_name="turbo", language="ko"):
    """
    Whisper로 음성을 텍스트로 변환합니다. (텍스트 추출만, metrics 계산 없음)
    
    Returns:
        {
            "text": "전체 텍스트",
            "segments": [
                {"id": 0, "text": "문장", "start": 0.0, "end": 5.2, "words": [...]},
                ...
            ],
            "duration": 120.5
        }
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    
    # segments를 간단한 형태로 변환 (metrics 없이)
    segments = []
    for seg in result["segments"]:
        segment_info = {
            "id": seg["id"],
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "words": []
        }
        
        # words 정보 포함 (시간 정보만)
        if "words" in seg:
            for w in seg["words"]:
                segment_info["words"].append({
                    "text": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })
        
        segments.append(segment_info)
    
    # duration 계산
    y, sr = librosa.load(audio_path, sr=16000)
    duration = float(len(y) / sr)
    
    return {"text": result["text"], "segments": segments, "duration": duration}


def analyze_segment_audio(audio_path: str, start_time: float, end_time: float, text: str = "", y=None, sr=None):
    """
    특정 시간 구간의 오디오를 librosa로 분석하여 metrics를 계산합니다.
    
    Args:
        audio_path: 오디오 파일 경로 (y가 None일 때만 사용)
        start_time: 시작 시간 (초)
        end_time: 끝 시간 (초)
        text: 해당 구간의 텍스트 (WPM 계산용)
        y: 오디오 데이터 (이미 로드된 경우 재사용)
        sr: 샘플링 레이트 (y가 제공된 경우 필수)
    
    Returns:
        {
            "dB": -12.5,
            "pitch_mean_hz": 180.0,
            "rate_wpm": 150.0,
            "pause_ratio": 0.15,
            "prosody_score": 85.5,
            "duration_sec": 2.5  # 단어 분석 시에만 포함
        }
    """
    # 오디오가 이미 로드되어 있으면 재사용, 없으면 로드
    if y is None or sr is None:
        y, sr = librosa.load(audio_path, sr=16000)
    
    start_samp = int(start_time * sr)
    end_samp = int(end_time * sr)
    y_seg = y[start_samp:end_samp]
    
    metrics = {}
    
    if len(y_seg) > 0:
        # dB 계산
        rms = librosa.feature.rms(y=y_seg)
        db = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))
        metrics["dB"] = round(db, 2)
        
        # Pitch 계산
        f0, _, _ = librosa.pyin(
            y_seg, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr
        )
        pitch_vals = f0[~np.isnan(f0)]
        metrics["pitch_mean_hz"] = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        
        # WPM 계산
        if text:
            words_count = len(text.split())
            duration_min = (end_time - start_time) / 60
            metrics["rate_wpm"] = words_count / duration_min if duration_min > 0 else 0.0
        else:
            metrics["rate_wpm"] = 0.0
        
        # Pause ratio 계산
        silence = np.sum(np.abs(y_seg) < 1e-4)
        metrics["pause_ratio"] = silence / len(y_seg) if len(y_seg) > 0 else 0.0
        
        # Prosody score 계산
        metrics["prosody_score"] = round(metrics["pitch_mean_hz"] * (1 - metrics["pause_ratio"]), 2)
        
        # Duration 계산 (단어 분석 시에만 유용)
        metrics["duration_sec"] = round(end_time - start_time, 3)
    else:
        metrics = {
            "dB": 0.0,
            "pitch_mean_hz": 0.0,
            "rate_wpm": 0.0,
            "pause_ratio": 0.0,
            "prosody_score": 0.0,
            "duration_sec": 0.0
        }
    
    return metrics


def analyze_segments(audio_path: str, model_name="turbo", language="ko"):
    """
    기존 호환성을 위한 함수 (재녹음 등에서 사용)
    Whisper 분석 + librosa 분석을 함께 수행합니다.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    y, sr = librosa.load(audio_path, sr=16000)

    analyzed = []
    for seg in result["segments"]:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_text = seg["text"].strip()
        metrics = {}

        start_samp, end_samp = int(seg_start*sr), int(seg_end*sr)
        y_seg = y[start_samp:end_samp]

        if len(y_seg) > 0:
            rms = librosa.feature.rms(y=y_seg)
            db = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))
            metrics["dB"] = round(db, 2)

            f0, _, _ = librosa.pyin(
                y_seg, fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"), sr=sr
            )
            pitch_vals = f0[~np.isnan(f0)]
            metrics["pitch_mean_hz"] = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0

            words_count = len(seg_text.split())
            duration_min = (seg_end - seg_start) / 60
            metrics["rate_wpm"] = words_count / duration_min if duration_min > 0 else 0

            silence = np.sum(np.abs(y_seg) < 1e-4)
            metrics["pause_ratio"] = silence / len(y_seg) if len(y_seg) > 0 else 0
            metrics["prosody_score"] = round(metrics["pitch_mean_hz"] * (1 - metrics["pause_ratio"]), 2)

        segment_info = {
            "id": seg["id"],
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "metrics": metrics,
            "words": []
        }

        if "words" in seg:
            for w in seg["words"]:
                w_text = w["word"].strip()
                w_start, w_end = w["start"], w["end"]
                w_start_samp, w_end_samp = int(w_start*sr), int(w_end*sr)
                y_word = y[w_start_samp:w_end_samp]

                if len(y_word) == 0:
                    continue

                # --- dB
                rms = librosa.feature.rms(y=y_word)
                db = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))

                # --- pitch
                f0, _, _ = librosa.pyin(
                    y_word,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr
                )
                pitch_vals = f0[~np.isnan(f0)]
                pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
                pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0

                duration = w_end - w_start

                segment_info["words"].append({
                    "text": w_text,
                    "start": w_start,
                    "end": w_end,
                    "metrics": {
                        "dB": round(db, 2),
                        "pitch_mean_hz": round(pitch_mean, 2),
                        "pitch_std_hz": round(pitch_std, 2),
                        "duration_sec": round(duration, 3)
                    }
                })

        analyzed.append(segment_info)

    return {"text": result["text"], "segments": analyzed, "duration": float(len(y) / sr)}
