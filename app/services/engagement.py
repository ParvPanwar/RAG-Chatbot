from typing import Optional


def calculate_engagement_rate(
    views: Optional[int],
    likes: Optional[int],
    comments: Optional[int],
) -> Optional[float]:
    if not views:
        return None

    total_engagements = (likes or 0) + (comments or 0)
    return round((total_engagements / views) * 100, 4)
