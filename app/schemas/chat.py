from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    analysis_id: str
    message: str = Field(..., min_length=1)


class Citation(BaseModel):
    citation: str
    video_id: str
    title: Optional[str] = None
    creator: Optional[str] = None
    chunk_id: str
    chunk_text: str
    start_time: Optional[float] = None   # seconds — present when chunk has timing
    end_time: Optional[float] = None     # seconds — present when chunk has timing


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
