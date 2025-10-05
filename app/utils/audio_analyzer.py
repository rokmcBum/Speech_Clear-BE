# app/utils/audio_analyzer.py
import whisper
import librosa
import numpy as np


def analyze_segments(audio_path: str, model_name="turbo", language="ko"):
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

        analyzed.append({
            "id": seg["id"],
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "metrics": metrics
        })

    return {"text": result["text"], "segments": analyzed, "duration": float(len(y) / sr)}
