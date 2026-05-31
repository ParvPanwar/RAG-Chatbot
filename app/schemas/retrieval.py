from typing import Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    analysis_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_text: str
    chunk_id: str
    video_id: str


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
