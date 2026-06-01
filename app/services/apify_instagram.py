import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.config import get_settings
from app.schemas.common import VideoMetadata

logger = logging.getLogger(__name__)

APIFY_RUN_URL = "https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"


class ApifyInstagramError(ValueError):
    pass


def fetch_reel_metadata(url: str) -> Optional[VideoMetadata]:
    settings = get_settings()
    token = settings.apify_api_token.strip()
    actor_id = settings.apify_instagram_actor_id.strip()

    if not token or not actor_id:
        logger.info("Apify Instagram metadata is not configured; using yt-dlp fallback.")
        return None

    actor_url = APIFY_RUN_URL.format(actor_id=normalize_actor_id(actor_id))
    payload = {
        "directUrls": [url],
        "resultsLimit": 1,
        "resultsType": "posts",
        "searchLimit": 1,
        "addParentData": False,
    }

    try:
        with httpx.Client(timeout=settings.apify_timeout_seconds) as client:
            response = client.post(actor_url, params={"token": token}, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as err:
        raise ApifyInstagramError("Apify Instagram metadata request failed.") from err

    items = response.json()
    if not isinstance(items, list) or not items:
        raise ApifyInstagramError("Apify did not return Instagram metadata.")

    if not isinstance(items[0], dict):
        raise ApifyInstagramError("Apify returned Instagram metadata in an unexpected format.")

    logger.info("Apify Instagram metadata returned %s item(s).", len(items))
    return make_metadata(items[0])


def make_metadata(item: dict[str, Any]) -> VideoMetadata:
    caption = pick_first_text(
        deep_get(item, "caption"),
        deep_get(item, "captionText"),
        deep_get(item, "text"),
        deep_get(item, "description"),
        deep_get(item, "title"),
    )
    creator = pick_first_text(
        deep_get(item, "ownerUsername"),
        deep_get(item, "username"),
        deep_get(item, "userUsername"),
        deep_get(item, "owner.username"),
        deep_get(item, "profile.username"),
        deep_get(item, "ownerFullName"),
        deep_get(item, "profileName"),
    )

    return VideoMetadata(
        title=pick_first_text(deep_get(item, "title"), caption, f"Video by {creator}" if creator else None),
        creator=creator or "Unknown Creator",
        follower_count=pick_first_int(
            deep_get(item, "ownerFollowersCount"),
            deep_get(item, "followersCount"),
            deep_get(item, "followerCount"),
            deep_get(item, "owner.followersCount"),
            deep_get(item, "profile.followersCount"),
        ),
        hashtags=extract_hashtags(item, caption),
        views=pick_first_int(
            deep_get(item, "videoViewCount"),
            deep_get(item, "videoPlayCount"),
            deep_get(item, "playCount"),
            deep_get(item, "playsCount"),
            deep_get(item, "viewCount"),
            deep_get(item, "viewsCount"),
            deep_get(item, "views_count"),
            deep_get(item, "views"),
            deep_get(item, "videoViews"),
            deep_get(item, "video_view_count"),
            deep_get(item, "video_play_count"),
            find_metric(item, ("video", "view")),
            find_metric(item, ("video", "play")),
            find_metric(item, ("view",)),
            find_metric(item, ("play",)),
        ),
        likes=pick_first_int(
            deep_get(item, "likesCount"),
            deep_get(item, "likeCount"),
            deep_get(item, "likes"),
        ),
        comments=pick_first_int(
            deep_get(item, "commentsCount"),
            deep_get(item, "commentCount"),
            deep_get(item, "comments"),
        ),
        upload_date=normalize_date(
            pick_first_text(
                deep_get(item, "timestamp"),
                deep_get(item, "takenAtTimestamp"),
                deep_get(item, "date"),
                deep_get(item, "uploadDate"),
                deep_get(item, "taken_at_timestamp"),
            )
        ),
        duration=pick_first_float(
            deep_get(item, "duration"),
            deep_get(item, "videoDuration"),
            deep_get(item, "videoDurationSeconds"),
        ),
    )


def normalize_actor_id(actor_id: str) -> str:
    return actor_id.strip().replace("/", "~")


def deep_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def pick_first_text(*values: Any) -> Optional[str]:
    for raw_value in values:
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if text:
            return text
    return None


def pick_first_int(*values: Any) -> Optional[int]:
    for raw_value in values:
        if isinstance(raw_value, bool) or raw_value is None:
            continue
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            return int(raw_value)
        if isinstance(raw_value, str):
            cleaned = raw_value.strip().replace(",", "")
            if cleaned.isdigit():
                return int(cleaned)
    return None


def pick_first_float(*values: Any) -> Optional[float]:
    for raw_value in values:
        if isinstance(raw_value, bool) or raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value.strip())
            except ValueError:
                continue
    return None


def find_metric(data: Any, name_parts: tuple[str, ...]) -> Optional[int]:
    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if all(part in normalized_key for part in name_parts):
                metric = pick_first_int(value)
                if metric is not None:
                    return metric

        for value in data.values():
            metric = find_metric(value, name_parts)
            if metric is not None:
                return metric

    if isinstance(data, list):
        for value in data:
            metric = find_metric(value, name_parts)
            if metric is not None:
                return metric

    return None


def extract_hashtags(item: dict[str, Any], caption: Optional[str]) -> list[str]:
    raw_tags: list[Any] = []
    for field in ("hashtags", "tags"):
        tags = item.get(field)
        if isinstance(tags, list):
            raw_tags.extend(tags)

    raw_tags.extend(re.findall(r"#([\w.-]+)", caption or ""))

    seen = set()
    hashtags = []
    for raw_tag in raw_tags:
        cleaned = str(raw_tag).strip().lstrip("#")
        if not cleaned:
            continue
        tag = f"#{cleaned}"
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        hashtags.append(tag)
    return hashtags[:30]


def normalize_date(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None

    if raw_value.isdigit():
        timestamp = int(raw_value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp // 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()

    return raw_value[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", raw_value) else raw_value
