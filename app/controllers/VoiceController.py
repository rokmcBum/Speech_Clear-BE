from fastapi import APIRouter, UploadFile, File
from app.service.VoiceService import process_voice

router = APIRouter(prefix="/voice", tags=["Voice"])

@router.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """
    음성 파일 업로드 후 분석 → JSON 결과 리턴
    """
    result = process_voice(file)
    return result
