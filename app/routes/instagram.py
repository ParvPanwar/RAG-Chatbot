from fastapi import APIRouter, HTTPException, status

from app.schemas.instagram import InstagramAnalyzeRequest, InstagramAnalyzeResponse
from app.services.instagram_analyzer import (
    InstagramAnalysisError,
    analyze_instagram_reel,
)

router = APIRouter(prefix="/analyze", tags=["instagram"])


@router.post("/instagram", response_model=InstagramAnalyzeResponse)
def analyze_instagram(request: InstagramAnalyzeRequest) -> InstagramAnalyzeResponse:
    try:
        return analyze_instagram_reel(str(request.url))
    except InstagramAnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
