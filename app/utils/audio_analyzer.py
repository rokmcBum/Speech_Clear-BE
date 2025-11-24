# app/utils/audio_analyzer.py
import whisper
import librosa
import numpy as np

# 한국어 문장 분할 라이브러리 (선택적)
try:
    import kss
    KSS_AVAILABLE = True
except ImportError:
    KSS_AVAILABLE = False
    print("⚠️ kss 라이브러리가 설치되지 않았습니다. 한국어 문장 분할 후처리를 건너뜁니다.")


def transcribe_audio(audio_path: str, model_name="turbo", language="ko"):
    """
    Whisper로 음성을 텍스트로 변환합니다. (텍스트 추출만, metrics 계산 없음)
    문장 분할 성능 향상을 위한 파라미터 최적화 적용
    
    Args:
        audio_path: 오디오 파일 경로
        model_name: Whisper 모델 이름 (turbo, base, small, medium, large-v3 등)
        language: 언어 코드 (ko: 한국어)
    
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
    
    # 문장 분할 성능 향상을 위한 파라미터 설정
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        # 이전 텍스트에 의존하지 않아 더 독립적인 문장 분할
        condition_on_previous_text=False,
        # 더 일관된 결과를 위한 낮은 temperature
        temperature=0,
        # 더 정확한 디코딩을 위한 beam_size
        beam_size=5,
        # 최적의 후보 선택
        best_of=5,
        # 더 세밀한 문장 분할을 위한 패턴 인식
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
    
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
    
    # 한국어 문장 분할 후처리 (kss 사용)
    if language == "ko" and KSS_AVAILABLE:
        segments = _refine_korean_segments(segments)
    
    # duration 계산
    y, sr = librosa.load(audio_path, sr=16000)
    duration = float(len(y) / sr)
    
    return {"text": result["text"], "segments": segments, "duration": duration}


def _refine_korean_segments(segments: list) -> list:
    """
    한국어 문장 분할을 개선하기 위한 후처리 함수.
    kss를 사용하여 Whisper segments를 더 정확하게 분할합니다.
    
    Args:
        segments: Whisper segments 리스트
    
    Returns:
        개선된 segments 리스트
    """
    if not segments:
        return segments
    
    refined_segments = []
    
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        
        # kss로 문장 분할
        try:
            sentences = kss.split_sentences(text)
        except Exception as e:
            # kss 분할 실패 시 원본 사용
            print(f"⚠️ kss 문장 분할 실패: {e}, 원본 segment 사용")
            refined_segments.append(seg)
            continue
        
        # 여러 문장으로 분할된 경우 시간을 비례적으로 분배
        if len(sentences) > 1:
            total_duration = seg["end"] - seg["start"]
            total_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
            
            current_time = seg["start"]
            accumulated_chars = 0
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # 문장의 문자 수 계산 (공백 제외)
                sentence_chars = len(sentence.replace(" ", "").replace("\n", "").replace("\t", ""))
                
                # 문장 길이에 비례하여 시간 계산
                if total_chars > 0:
                    sentence_ratio = sentence_chars / total_chars
                    sentence_duration = total_duration * sentence_ratio
                else:
                    sentence_duration = total_duration / len(sentences)
                
                # 마지막 문장은 남은 시간 모두 사용
                if i == len(sentences) - 1:
                    sentence_end = seg["end"]
                else:
                    sentence_end = current_time + sentence_duration
                
                # words를 문장 시간 범위에 맞게 필터링
                sentence_words = []
                if seg.get("words"):
                    for word in seg["words"]:
                        word_start = word.get("start", 0)
                        word_end = word.get("end", 0)
                        # 문장 시간 범위 내의 words만 포함
                        if word_start >= current_time and word_end <= sentence_end:
                            sentence_words.append(word)
                
                refined_segments.append({
                    "id": seg["id"] if i == 0 else f"{seg['id']}_{i}",
                    "text": sentence,
                    "start": current_time,
                    "end": sentence_end,
                    "words": sentence_words
                })
                
                current_time = sentence_end
                accumulated_chars += sentence_chars
        else:
            # 문장이 하나면 원본 그대로 사용
            refined_segments.append(seg)
    
    return refined_segments


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
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0,
        beam_size=5,
        best_of=5,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
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
