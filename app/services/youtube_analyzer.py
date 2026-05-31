import logging
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.schemas.youtube import (
    TranscriptSegment,
    YouTubeAnalyzeResponse,
    VideoMetadata,
)
from app.services.engagement import calculate_engagement_rate

logger = logging.getLogger(__name__)


class YouTubeAnalysisError(ValueError):
    pass


def analyze_youtube_video(url: str, assigned_id: str = None) -> YouTubeAnalyzeResponse:
    video_id = extract_video_id(url)
    logger.info(f"Starting analysis of YouTube URL: {url} (ID: {video_id})")
    
    metadata = fetch_metadata(url)
    logger.info(f"YouTube metadata loaded: '{metadata.title}' by {metadata.creator}")
    
    try:
        logger.info(f"Attempting to fetch official captions for YouTube ID {video_id}...")
        transcript = fetch_transcript(video_id)
        logger.info(f"Successfully retrieved official captions ({len(transcript)} segments)")
    except YouTubeAnalysisError as exc:
        logger.warning(
            f"Official YouTube caption API failed for ID {video_id}: {exc}. "
            "Attempting Whisper audio download and transcription fallback..."
        )
        try:
            from tempfile import TemporaryDirectory
            from pathlib import Path
            from app.services.instagram_analyzer import transcribe_audio
            
            with TemporaryDirectory(prefix="youtube-audio-") as temp_dir:
                logger.info(f"Downloading YouTube audio to {temp_dir} using yt-dlp...")
                audio_path = _download_youtube_audio(url=url, output_dir=Path(temp_dir))
                logger.info("Transcribing downloaded YouTube audio with Whisper model...")
                transcript = transcribe_audio(audio_path)
            
            logger.info(f"Whisper fallback successfully generated {len(transcript)} transcript segments!")
        except Exception as fallback_exc:
            logger.error(f"Whisper fallback failed for YouTube ID {video_id}: {fallback_exc}", exc_info=True)
            raise YouTubeAnalysisError(
                f"Transcript is unavailable. Caption API and audio transcription both failed: {fallback_exc}"
            ) from fallback_exc

    return YouTubeAnalyzeResponse(
        platform="youtube",
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


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            path_parts = parsed.path.strip("/").split("/")
            video_id = path_parts[1] if len(path_parts) > 1 else ""
        else:
            video_id = ""
    else:
        video_id = ""

    if not video_id:
        logger.warning(f"Failed to extract video ID from invalid YouTube URL: {url}")
        raise YouTubeAnalysisError("Invalid YouTube URL.")

    return video_id


def _download_youtube_audio(url: str, output_dir) -> "Path":
    """Download YouTube audio using yt-dlp and return the path to the audio file."""
    from pathlib import Path
    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
    }
    try:
        with YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        logger.error(f"yt-dlp YouTube audio download failed for {url}: {exc}", exc_info=True)
        raise YouTubeAnalysisError("Unable to download YouTube audio for transcription.") from exc

    output_dir = Path(output_dir)
    expected_path = output_dir / f"{info.get('id')}.{info.get('ext')}"
    if expected_path.exists():
        return expected_path

    downloaded_files = [p for p in output_dir.iterdir() if p.is_file()]
    if not downloaded_files:
        raise YouTubeAnalysisError("YouTube audio download did not produce a file.")
    return max(downloaded_files, key=lambda p: p.stat().st_size)


def fetch_transcript(video_id: str) -> list[TranscriptSegment]:
    """Fetch YouTube transcript in any available language.

    Strategy:
    1. Try English (en) first.
    2. Fall back to any manually-created transcript.
    3. Fall back to any auto-generated transcript.
    This ensures non-English videos (e.g. Hindi auto-captions) are not rejected.
    """
    api = YouTubeTranscriptApi()

    try:
        # Step 1: try English
        fetched = api.fetch(video_id, languages=["en"])
        logger.info(f"Fetched English transcript for {video_id}")
    except Exception:
        # Step 2 & 3: pick any available transcript
        try:
            transcript_list = api.list(video_id)
            # Prefer manual transcripts, then auto-generated
            all_transcripts = list(transcript_list)
            manual = [t for t in all_transcripts if not t.is_generated]
            auto   = [t for t in all_transcripts if t.is_generated]
            chosen = (manual or auto)
            if not chosen:
                raise YouTubeAnalysisError("No transcripts available for this video.")
            selected = chosen[0]
            logger.info(
                f"No English transcript for {video_id}; "
                f"using '{selected.language}' ({selected.language_code}, "
                f"auto={selected.is_generated})"
            )
            fetched = api.fetch(video_id, languages=[selected.language_code])
        except YouTubeAnalysisError:
            raise
        except Exception as exc:
            raise YouTubeAnalysisError("Unable to retrieve transcript.") from exc

    segments = []
    for snippet in fetched.snippets:
        try:
            start = float(snippet.start)
        except (ValueError, TypeError):
            start = 0.0
        try:
            duration = float(snippet.duration)
        except (ValueError, TypeError):
            duration = 0.0
        text = (snippet.text or "").strip()
        if text:
            segments.append(TranscriptSegment(text=text, start=start, duration=duration))

    if not segments:
        raise YouTubeAnalysisError("Transcript is empty or not available for this video.")

    return segments


def fetch_metadata(url: str) -> VideoMetadata:
    ydl_options = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,
        "noplaylist": True,
        "check_formats": False,
        "ignoreerrors": True,
    }

    try:
        with YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        logger.error(f"yt-dlp metadata download failed for {url}: {exc}", exc_info=True)
        raise YouTubeAnalysisError("Unable to retrieve YouTube metadata.") from exc

    if not info:
        logger.warning(f"yt-dlp extracted no info for {url}.")
        raise YouTubeAnalysisError("Unable to retrieve YouTube metadata.")

    creator = first_text(
        info.get("uploader"),
        info.get("channel"),
        info.get("creator"),
        info.get("uploader_id"),
        info.get("channel_id"),
        fetch_oembed_author(url),
        extract_handle(info.get("channel_url")),
        extract_handle(info.get("uploader_url")),
    )

    return VideoMetadata(
        title=info.get("title") or "YouTube Video",
        creator=creator,
        follower_count=first_int(
            info.get("channel_follower_count"),
            info.get("uploader_follower_count"),
            info.get("subscriber_count"),
            info.get("channel_subscriber_count"),
        ),
        hashtags=extract_hashtags(info),
        views=info.get("view_count"),
        likes=info.get("like_count"),
        comments=info.get("comment_count"),
        upload_date=info.get("upload_date"),
        duration=first_float(info.get("duration")),
    )


def first_text(*values) -> Optional[str]:
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if text and text.upper() != "NA":
            return text
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


def fetch_oembed_author(url: str) -> Optional[str]:
    try:
        response = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=8,
        )
        if response.status_code != 200:
            return None
        return first_text(response.json().get("author_name"))
    except Exception:
        return None


def extract_handle(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path.startswith("@"):
        return path
    return None


def extract_hashtags(info: dict) -> list[str]:
    tags = info.get("tags") or []
    hashtags = []
    for tag in tags:
        normalized = normalize_hashtag(str(tag))
        if normalized:
            hashtags.append(normalized)

    description = info.get("description") or ""
    hashtags.extend(normalize_hashtag(match) for match in re.findall(r"#([\w.-]+)", description))
    return dedupe_hashtags(hashtags)


def normalize_hashtag(value: str) -> str:
    cleaned = value.strip().lstrip("#")
    if not cleaned:
        return ""
    return f"#{cleaned}"


def dedupe_hashtags(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped[:30]
