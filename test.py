# import whisper
#
# model = whisper.load_model("turbo")
#
# # 30초 길이에 맞게 오디오 로드 & 패딩/트리밍
# audio = whisper.load_audio("voice1.m4a")
# audio = whisper.pad_or_trim(audio)
#
# # Mel-spectrogram 변환 후 모델과 같은 디바이스로 이동
# mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
#
# # 언어 감지
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
#
# # 디코딩
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)
#
# # 결과 출력
# print(result.text)

import whisper

model = whisper.load_model("turbo")  # turbo 말고 medium/large 추천 (한국어 정확도 ↑)

result = model.transcribe("voice1.m4a", language="ko")

print("=== 전체 텍스트 ===")
print(result["text"])

print("\n=== 문장 단위(segment) ===")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} ~ {seg['end']:.2f}] {seg['text']}")
