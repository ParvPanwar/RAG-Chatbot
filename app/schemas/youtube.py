from pydantic import BaseModel, Field, HttpUrl

from app.schemas.common import TranscriptSegment, VideoMetadata
from app.schemas.common import VideoAnalysis as YouTubeAnalyzeResponse

class YouTubeAnalyzeRequest(BaseModel):
    url: HttpUrl = Field(..., description="A public YouTube video URL.")
