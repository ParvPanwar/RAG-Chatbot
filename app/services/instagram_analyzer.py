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

logger = logging.getLogger(__name__)

INSTAGRAM_REEL_PATTERN = re.compile(
    r"instagram\.com/(?:reel|p|tv)/(?P<id>[^/?#]+)",
    re.IGNORECASE,
)


class InstagramAnalysisError(ValueError):
    pass


def analyze_instagram_reel(url: str, assigned_id: str = None) -> VideoAnalysis:
    video_id = extract_instagram_video_id(url)
    logger.info(f"Starting analysis of Instagram Reel URL: {url} (ID: {video_id})")
    
    info = fetch_instagram_info(url)
    metadata = build_metadata(info)
    logger.info(f"Instagram metadata loaded: '{metadata.title}' by {metadata.creator}")

    with TemporaryDirectory(prefix="instagram-audio-") as temp_dir:
        logger.info(f"Downloading Instagram audio to {temp_dir} using yt-dlp...")
        audio_path = download_audio(url=url, output_dir=Path(temp_dir))
        logger.info("Transcribing Instagram Reel audio with Whisper model...")
        transcript = transcribe_audio(audio_path)

    return VideoAnalysis(
        platform="instagram",
        video_id=video_id,
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


def extract_instagram_video_id(url: str) -> str:
    match = INSTAGRAM_REEL_PATTERN.search(url)
    if not match:
        logger.warning(f"Failed to extract Reel ID from invalid Instagram URL: {url}")
        raise InstagramAnalysisError("Invalid Instagram Reel URL.")
    return match.group("id")


def fetch_instagram_info(url: str) -> dict:
    ydl_options = instagram_ydl_options({
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "no_warnings": True,
    })

    try:
        with YoutubeDL(ydl_options) as ydl:
            return ydl.extract_info(url, download=False)
    except DownloadError as exc:
        logger.error(f"yt-dlp Instagram metadata download failed for {url}: {exc}", exc_info=True)
        raise InstagramAnalysisError("Unable to retrieve Instagram metadata.") from exc


def build_metadata(info: dict) -> VideoMetadata:
    return VideoMetadata(
        title=info.get("title") or info.get("description"),
        creator=info.get("uploader") or info.get("channel") or "Unknown Creator",
        follower_count=first_int(
            info.get("uploader_follower_count"),
            info.get("channel_follower_count"),
            info.get("creator_follower_count"),
            info.get("subscriber_count"),
        ),
        hashtags=extract_hashtags(info),
        views=first_int(
            info.get("view_count"),
            info.get("play_count"),
            info.get("video_view_count"),
            info.get("view_count_reel"),
        ),
        likes=first_int(info.get("like_count")),
        comments=first_int(info.get("comment_count")),
        upload_date=info.get("upload_date"),
        duration=first_float(info.get("duration")),
    )


def first_float(*values) -> Optional[float]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def first_int(*values) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def extract_hashtags(info: dict) -> list[str]:
    values = []
    values.extend(info.get("tags") or [])
    description = info.get("description") or info.get("title") or ""
    values.extend(re.findall(r"#([\w.-]+)", description))

    seen = set()
    hashtags = []
    for value in values:
        cleaned = str(value).strip().lstrip("#")
        if not cleaned:
            continue
        tag = f"#{cleaned}"
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        hashtags.append(tag)
    return hashtags[:30]


def download_audio(url: str, output_dir: Path) -> Path:
    ydl_options = instagram_ydl_options({
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
    })

    try:
        with YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        logger.error(f"yt-dlp Instagram audio download failed for {url}: {exc}", exc_info=True)
        raise InstagramAnalysisError("Unable to download Instagram Reel audio.") from exc

    expected_path = output_dir / f"{info.get('id')}.{info.get('ext')}"
    if expected_path.exists():
        return expected_path

    downloaded_files = [path for path in output_dir.iterdir() if path.is_file()]
    if not downloaded_files:
        raise InstagramAnalysisError("Instagram audio download did not produce a file.")

    return max(downloaded_files, key=lambda path: path.stat().st_size)


def instagram_ydl_options(base_options: dict) -> dict:
    options = dict(base_options)
    cookie_file = get_settings().instagram_cookie_file.strip()
    if cookie_file:
        path = Path(cookie_file).expanduser()
        if not path.exists():
            raise InstagramAnalysisError(
                f"Instagram cookie file was configured but not found: {path}"
            )
        options["cookiefile"] = str(path)
    return options


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
