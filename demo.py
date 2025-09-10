import whisper
import librosa
import numpy as np

def analyze_segments(audio_path: str, model_name="turbo", language="ko"):
    # 1. Whisper 모델 로드
    model = whisper.load_model(model_name)

    # 2. 전사 (segment + word timestamps)
    result = model.transcribe(audio_path, language=language, word_timestamps=True)

    # 3. librosa로 waveform 로드 (한 번만!)
    y, sr = librosa.load(audio_path, sr=16000)

    analyzed = []

    for seg in result["segments"]:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_text = seg["text"].strip()

        segment_info = {
            "sid": seg["id"],
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "metrics": {},
            "words": []
        }

        # -------------------
        # Segment-level 분석
        # -------------------
        start_samp, end_samp = int(seg_start*sr), int(seg_end*sr)
        y_seg = y[start_samp:end_samp]

        if len(y_seg) > 0:
            # --- dB (RMS → dB 변환)
            rms = librosa.feature.rms(y=y_seg)
            db = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))

            # --- pitch (pyin 사용)
            f0, _, _ = librosa.pyin(
                y_seg,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )
            pitch_vals = f0[~np.isnan(f0)]
            pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
            pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0

            # --- rate (WPM)
            words_count = len(seg_text.split())
            duration_min = (seg_end - seg_start) / 60
            wpm = words_count / duration_min if duration_min > 0 else 0

            # --- prosody (간단 지표: pitch_std × (1 - pause_ratio))
            silence = np.sum(np.abs(y_seg) < 1e-4)
            pause_ratio = silence / len(y_seg) if len(y_seg) > 0 else 0
            prosody_score = pitch_std * (1 - pause_ratio)

            segment_info["metrics"] = {
                "dB": round(db, 2),
                "pitch_mean_hz": round(pitch_mean, 2),
                "pitch_std_hz": round(pitch_std, 2),
                "rate_wpm": round(wpm, 1),
                "pause_ratio": round(pause_ratio, 3),
                "prosody_score": round(prosody_score, 2)
            }

        # -------------------
        # Word-level 분석
        # -------------------
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

    return {
        "text": result["text"],
        "segments": analyzed
    }


# 사용 예시
if __name__ == "__main__":
    data = analyze_segments("voice.m4a", model_name="turbo", language="ko")
    import json
    print(json.dumps(data, ensure_ascii=False, indent=2))
