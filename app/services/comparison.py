from app.schemas.common import VideoAnalysis
from app.schemas.comparison import ComparisonSummary


def build_comparison_summary(
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> ComparisonSummary:
    rate_a = video_a.engagement_rate
    rate_b = video_b.engagement_rate

    higher_video = None
    rate_gap = None
    if rate_a is not None and rate_b is not None:
        rate_gap = round(abs(rate_a - rate_b), 4)
        if rate_a > rate_b:
            higher_video = "A"
        elif rate_b > rate_a:
            higher_video = "B"
        else:
            higher_video = "tie"

    duration_difference = None
    if video_a.metadata.duration is not None and video_b.metadata.duration is not None:
        duration_difference = round(abs(video_a.metadata.duration - video_b.metadata.duration), 3)

    return ComparisonSummary(
        higher_engagement_platform=higher_video,
        engagement_rate_gap=rate_gap,
        duration_difference_seconds=duration_difference,
    )
