from pydantic import BaseModel, Field, HttpUrl

from app.schemas.common import VideoAnalysis as InstagramAnalyzeResponse


class InstagramAnalyzeRequest(BaseModel):
    url: HttpUrl = Field(..., description="A public Instagram Reel URL.")
