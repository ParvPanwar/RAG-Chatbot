from typing import Optional

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    title: Optional[str] = None
    creator: Optional[str] = None
    follower_count: Optional[int] = Field(
        None,
        description="Subscriber/follower count for the creator when available.",
    )
    hashtags: list[str] = Field(default_factory=list)
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    upload_date: Optional[str] = None
    duration: Optional[float] = Field(None, description="Video duration in seconds.")


class TranscriptSegment(BaseModel):
    text: str
    start: float = 0
    duration: float = 0


class VideoAnalysis(BaseModel):
    platform: str
    video_id: str
    video_label: Optional[str] = Field(
        None,
        description="Comparison label shown to users, usually A or B.",
    )
    url: str
    metadata: VideoMetadata
    transcript: list[TranscriptSegment]
    engagement_rate: Optional[float] = Field(
        None,
        description="Engagement rate as a percentage: ((likes + comments) / views) * 100.",
    )
