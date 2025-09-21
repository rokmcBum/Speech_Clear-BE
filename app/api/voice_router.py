from fastapi import APIRouter, UploadFile, File

from app.controllers.VoiceController import analyze

router = APIRouter(
    prefix="/voice",
    tags=["voice"]
)

@router.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """
    음성 파일 업로드 후 분석 → JSON 결과 리턴
    """
    result = await analyze(file)
    return result
