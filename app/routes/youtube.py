from fastapi import APIRouter, HTTPException, status

from app.schemas.youtube import YouTubeAnalyzeRequest, YouTubeAnalyzeResponse
from app.services.youtube_analyzer import YouTubeAnalysisError, analyze_youtube_video

router = APIRouter(prefix="/analyze", tags=["youtube"])


@router.post("/youtube", response_model=YouTubeAnalyzeResponse)
def analyze_youtube(request: YouTubeAnalyzeRequest) -> YouTubeAnalyzeResponse:
    try:
        return analyze_youtube_video(str(request.url))
    except YouTubeAnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
