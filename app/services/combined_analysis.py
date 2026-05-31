import logging
from app.schemas.analysis import AnalyzeResponse
from app.services.analysis_store import save_analysis_result
from app.services.instagram_analyzer import analyze_instagram_reel, InstagramAnalysisError
from app.services.youtube_analyzer import analyze_youtube_video, YouTubeAnalysisError
from app.services.comparison import build_comparison_summary

logger = logging.getLogger(__name__)


def analyze_videos(platform: str, video_url_a: str, video_url_b: str) -> AnalyzeResponse:
    logger.info(f"Initializing same-platform analysis workflow for platform: {platform}...")
    
    if platform == "youtube":
        try:
            # Validate early so the user gets one clear error for either URL.
            from app.services.youtube_analyzer import get_youtube_id
            get_youtube_id(video_url_a)
            get_youtube_id(video_url_b)
        except Exception as err:
            logger.warning(f"URL validation failed for YouTube inputs: {err}")
            raise YouTubeAnalysisError("Both input URLs must be valid YouTube URLs.") from err
            
        logger.info("Processing first YouTube video as Video A...")
        video_a = analyze_youtube_video(video_url_a, assigned_id="A")
        logger.info("Processing second YouTube video as Video B...")
        video_b = analyze_youtube_video(video_url_b, assigned_id="B")
        
    elif platform == "instagram":
        try:
            # Instagram URLs are fussier, so validate before downloading audio.
            from app.services.instagram_analyzer import get_reel_id
            get_reel_id(video_url_a)
            get_reel_id(video_url_b)
        except Exception as err:
            logger.warning(f"URL validation failed for Instagram inputs: {err}")
            raise InstagramAnalysisError("Both input URLs must be valid Instagram Reel URLs.") from err
            
        logger.info("Processing first Instagram Reel as Video A...")
        video_a = analyze_instagram_reel(video_url_a, assigned_id="A")
        logger.info("Processing second Instagram Reel as Video B...")
        video_b = analyze_instagram_reel(video_url_b, assigned_id="B")
        
    else:
        logger.error(f"Invalid platform specified: {platform}")
        raise ValueError("Unsupported platform. Please select either 'youtube' or 'instagram'.")
    
    logger.info("Generating platform comparison analytics...")
    comparison = build_comparison_summary(video_a, video_b)
    
    result = AnalyzeResponse(
        analysis_id="",
        video_a=video_a,
        video_b=video_b,
        comparison=comparison,
    )
    
    # Keep the latest analysis in memory for later requests.
    analysis_id = save_analysis_result(result)
    result.analysis_id = analysis_id
    logger.info(f"Analysis result saved with ID: {analysis_id}")
    
    return result
