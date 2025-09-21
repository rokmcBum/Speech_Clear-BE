from fastapi import UploadFile
from app.utils.audio_analyzer import analyze_segments


def process_voice(file: UploadFile):
    # 분석 실행
    result = analyze_segments(file.filename, model_name="turbo", language="ko")
    return result
