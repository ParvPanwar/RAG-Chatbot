from typing import Optional
from pydantic import BaseModel, HttpUrl

from app.schemas.common import VideoAnalysis
from app.schemas.comparison import ComparisonSummary


class AnalyzeRequest(BaseModel):
    platform: str
    video_url_a: HttpUrl
    video_url_b: HttpUrl


class AnalyzeResponse(BaseModel):
    analysis_id: str
    video_a: VideoAnalysis
    video_b: VideoAnalysis
    comparison: Optional[ComparisonSummary] = None
    chunks_indexed: Optional[int] = None
