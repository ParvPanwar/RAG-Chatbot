import logging
import re
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.config import get_settings
from app.schemas.common import TranscriptSegment, VideoAnalysis, VideoMetadata
from app.services.engagement import calculate_engagement_rate
from app.services.apify_instagram import ApifyInstagramError, fetch_reel_metadata

logger = logging.getLogger(__name__)

INSTAGRAM_REEL_PATTERN = re.compile(
    r"instagram\.com/(?:reel|p|tv)/(?P<id>[^/?#]+)",
    re.IGNORECASE,
)


class InstagramAnalysisError(ValueError):
    pass


def analyze_instagram_reel(url: str, assigned_id: str = None) -> VideoAnalysis:
    reel_id = get_reel_id(url)
    logger.info(f"Starting analysis of Instagram Reel URL: {url} (ID: {reel_id})")
    
    metadata = get_reel_metadata(url)
    logger.info(f"Instagram metadata loaded: '{metadata.title}' by {metadata.creator}")

    with TemporaryDirectory(prefix="instagram-audio-") as temp_dir:
        logger.info(f"Downloading Instagram audio to {temp_dir} using yt-dlp...")
        audio_path = download_reel_audio(url=url, out_dir=Path(temp_dir))
        logger.info("Transcribing Instagram Reel audio with Whisper model...")
        transcript = transcribe_audio(audio_path)

    return VideoAnalysis(
        platform="instagram",
        video_id=reel_id,
        video_label=assigned_id,
        url=url,
        metadata=metadata,
        transcript=transcript,
        engagement_rate=calculate_engagement_rate(
            views=metadata.views,
            likes=metadata.likes,
            comments=metadata.comments,
        ),
    )


def get_reel_id(url: str) -> str:
    """Pull the short Reel code out of a public Instagram URL."""
    match = INSTAGRAM_REEL_PATTERN.search(url)
    if not match:
        logger.warning(f"Failed to extract Reel ID from invalid Instagram URL: {url}")
        raise InstagramAnalysisError("Invalid Instagram Reel URL.")
    return match.group("id")


def get_reel_info(url: str) -> dict:
    """Read Reel metadata with yt-dlp; cookies are optional for local testing."""
    ydl_opts = add_instagram_cookies({
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "no_warnings": True,
    })

    try:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    except DownloadError as err:
        logger.error(f"yt-dlp Instagram metadata download failed for {url}: {err}", exc_info=True)
        raise InstagramAnalysisError("Unable to retrieve Instagram metadata.") from err


def get_reel_metadata(url: str) -> VideoMetadata:
    """Prefer Apify for public metrics, then fall back to yt-dlp."""
    # yt-dlp often misses Instagram Reel views, so Apify is used for public metrics.
    try:
        apify_metadata = fetch_reel_metadata(url)
    except ApifyInstagramError as err:
        logger.warning(f"Apify Instagram metadata failed; falling back to yt-dlp: {err}")
        apify_metadata = None

    if apify_metadata:
        logger.info(
            "Instagram metadata source: Apify (views=%s, likes=%s, comments=%s)",
            apify_metadata.views,
            apify_metadata.likes,
            apify_metadata.comments,
        )
        return apify_metadata

    reel_info = get_reel_info(url)
    metadata = make_reel_metadata(reel_info)
    logger.info(
        "Instagram metadata source: yt-dlp fallback (views=%s, likes=%s, comments=%s)",
        metadata.views,
        metadata.likes,
        metadata.comments,
    )
    return metadata


def make_reel_metadata(info: dict) -> VideoMetadata:
    """Normalize Instagram/yt-dlp's mixed field names into our app schema."""
    return VideoMetadata(
        title=info.get("title") or info.get("description"),
        creator=info.get("uploader") or info.get("channel") or "Unknown Creator",
        follower_count=pick_first_int(
            info.get("uploader_follower_count"),
            info.get("channel_follower_count"),
            info.get("creator_follower_count"),
            info.get("subscriber_count"),
        ),
        hashtags=extract_hashtags(info),
        views=pick_first_int(
            info.get("view_count"),
            info.get("play_count"),
            info.get("video_view_count"),
            info.get("view_count_reel"),
        ),
        likes=pick_first_int(info.get("like_count")),
        comments=pick_first_int(info.get("comment_count")),
        upload_date=info.get("upload_date"),
        duration=pick_first_float(info.get("duration")),
    )


def pick_first_float(*values) -> Optional[float]:
    """Return the first value that behaves like seconds with decimals."""
    for raw_value in values:
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except ValueError:
                continue
    return None


def pick_first_int(*values) -> Optional[int]:
    """Return the first value that can safely be shown as a count."""
    for raw_value in values:
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            return int(raw_value)
        if isinstance(raw_value, str) and raw_value.isdigit():
            return int(raw_value)
    return None


def extract_hashtags(info: dict) -> list[str]:
    """Collect hashtags from both structured tags and caption text."""
    raw_tags = []
    raw_tags.extend(info.get("tags") or [])
    description = info.get("description") or info.get("title") or ""
    raw_tags.extend(re.findall(r"#([\w.-]+)", description))

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


def download_reel_audio(url: str, out_dir: Path) -> Path:
    """Save Reel audio in a temp folder so Whisper can transcribe it."""
    ydl_opts = add_instagram_cookies({
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
    })

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as err:
        logger.error(f"yt-dlp Instagram audio download failed for {url}: {err}", exc_info=True)
        raise InstagramAnalysisError("Unable to download Instagram Reel audio.") from err

    expected_path = out_dir / f"{info.get('id')}.{info.get('ext')}"
    if expected_path.exists():
        return expected_path

    files = [path for path in out_dir.iterdir() if path.is_file()]
    if not files:
        raise InstagramAnalysisError("Instagram audio download did not produce a file.")

    return max(files, key=lambda path: path.stat().st_size)


def add_instagram_cookies(base_opts: dict) -> dict:
    """Attach a local cookies.txt file to yt-dlp options when configured."""
    opts = dict(base_opts)
    cookie_file = get_settings().instagram_cookie_file.strip()
    if cookie_file:
        path = Path(cookie_file).expanduser()
        if not path.exists():
            raise InstagramAnalysisError(
                f"Instagram cookie file was configured but not found: {path}"
            )
        opts["cookiefile"] = str(path)
    return opts


def transcribe_audio(audio_path: Path) -> list[TranscriptSegment]:
    try:
        segments, _ = get_whisper_model().transcribe(str(audio_path))
        transcript = [
            TranscriptSegment(
                text=segment.text.strip(),
                start=float(segment.start),
                duration=float(segment.end - segment.start),
            )
            for segment in segments
            if segment.text.strip()
        ]
    except Exception as exc:
        logger.error(f"Whisper transcription failed for {audio_path}: {exc}", exc_info=True)
        raise InstagramAnalysisError("Unable to transcribe Instagram Reel audio.") from exc

    if not transcript:
        logger.warning(f"No speech was detected in Instagram audio {audio_path}.")
        raise InstagramAnalysisError("No speech was detected in this Instagram Reel audio.")

    return transcript


@lru_cache
def get_whisper_model() -> WhisperModel:
    settings = get_settings()
    logger.info(
        f"Loading Whisper model (size: {settings.whisper_model_size}, "
        f"device: {settings.whisper_device}, compute_type: {settings.whisper_compute_type})..."
    )
    return WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
