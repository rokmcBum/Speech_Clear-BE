import tempfile
from fastapi import UploadFile
from app.utils.audio_analyzer import analyze_segments

def process_voice(file: UploadFile):
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # 분석 실행
    result = analyze_segments(tmp_path, model_name="turbo", language="ko")
    return result
