import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.combined_analysis import analyze_videos
from app.services.instagram_analyzer import InstagramAnalysisError
from app.services.youtube_analyzer import YouTubeAnalysisError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    logger.info(f"POST /analyze request received: platform='{request.platform}', url_a='{request.video_url_a}', url_b='{request.video_url_b}'")
    try:
        response = analyze_videos(
            platform=request.platform,
            video_url_a=str(request.video_url_a),
            video_url_b=str(request.video_url_b),
        )
        logger.info(f"POST /analyze success: analysis_id={response.analysis_id}")
        return response
    except (YouTubeAnalysisError, InstagramAnalysisError) as exc:
        logger.warning(f"POST /analyze failed with expected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"POST /analyze failed with unexpected error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during video analysis: {exc}",
        ) from exc
