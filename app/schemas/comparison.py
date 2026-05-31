from typing import Optional

from pydantic import BaseModel


class ComparisonSummary(BaseModel):
    higher_engagement_platform: Optional[str]
    engagement_rate_gap: Optional[float]
    duration_difference_seconds: Optional[float]
