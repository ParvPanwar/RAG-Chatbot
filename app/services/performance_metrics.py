from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Set

from app.schemas.common import TranscriptSegment, VideoAnalysis

# Words that indicate a call-to-action in the transcript text
_CTA_WORDS: Set[str] = {
    "subscribe",
    "follow",
    "like",
    "comment",
    "share",
    "click",
    "link",
    "bio",
    "tap",
    "check out",
    "join",
    "sign up",
    "download",
    "watch",
    "turn on",
    "notification",
}


def format_performance_metrics(video_a: VideoAnalysis, video_b: VideoAnalysis) -> str:
    """
    Return a multi-line string with all computed comparative metrics, ready to
    embed in a prompt.  Missing data is labelled explicitly so the LLM doesn't
    need to guess whether a field is absent.
    """
    lines: List[str] = ["=== Computed Performance Metrics ==="]

    # Engagement rates (already computed on VideoAnalysis)
    _append_engagement_section(lines, video_a, video_b)

    # Raw counts
    _append_count_section(lines, video_a, video_b)

    # Duration
    _append_duration_section(lines, video_a, video_b)

    # Upload date gap
    _append_date_section(lines, video_a, video_b)

    # Follower counts
    _append_follower_section(lines, video_a, video_b)

    # Hashtag overlap
    _append_hashtag_section(lines, video_a, video_b)

    # Title similarity
    _append_title_section(lines, video_a, video_b)

    # CTA presence
    _append_cta_section(lines, video_a, video_b)

    # Transcript pacing
    _append_pacing_section(lines, video_a, video_b)

    # Transcript energy (simple exclamation-ratio proxy)
    _append_energy_section(lines, video_a, video_b)

    lines.append("=" * 36)
    return "\n".join(lines)


def _append_engagement_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    rate_a = video_a.engagement_rate
    rate_b = video_b.engagement_rate
    lines.append("\n[Engagement Rates]")
    lines.append(f"  Video A engagement rate: {_fmt_pct(rate_a)}")
    lines.append(f"  Video B engagement rate: {_fmt_pct(rate_b)}")

    if rate_a is not None and rate_b is not None:
        gap = rate_a - rate_b
        winner = "A" if gap > 0 else ("B" if gap < 0 else "tie")
        lines.append(f"  Engagement rate gap: {abs(gap):.4f}% (Video {winner} leads)")
    else:
        lines.append("  Engagement rate gap: unavailable (one or both rates missing)")


def _append_count_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    meta_a, meta_b = video_a.metadata, video_b.metadata
    lines.append("\n[Raw Counts]")

    if meta_a.views is not None and meta_b.views is not None:
        view_gap = abs(meta_a.views - meta_b.views)
        more = "A" if meta_a.views > meta_b.views else ("B" if meta_b.views > meta_a.views else "tie")
        lines.append(f"  Video A views: {meta_a.views:,}")
        lines.append(f"  Video B views: {meta_b.views:,}")
        lines.append(f"  View gap: {view_gap:,} (Video {more} has more)")
    else:
        lines.append(f"  Video A views: {_fmt_int(meta_a.views)}")
        lines.append(f"  Video B views: {_fmt_int(meta_b.views)}")

    # Per-view rates keep big channels from winning only because they are big.
    like_a = _rate(meta_a.likes, meta_a.views)
    like_b = _rate(meta_b.likes, meta_b.views)
    lines.append(f"  Video A like rate: {_fmt_pct(like_a)}  (likes/views)")
    lines.append(f"  Video B like rate: {_fmt_pct(like_b)}  (likes/views)")

    comment_rate_a = _rate(meta_a.comments, meta_a.views)
    comment_rate_b = _rate(meta_b.comments, meta_b.views)
    lines.append(f"  Video A comment rate: {_fmt_pct(comment_rate_a)}  (comments/views)")
    lines.append(f"  Video B comment rate: {_fmt_pct(comment_rate_b)}  (comments/views)")


def _append_duration_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    dur_a, dur_b = video_a.metadata.duration, video_b.metadata.duration
    lines.append("\n[Duration]")
    lines.append(f"  Video A duration: {_fmt_dur(dur_a)}")
    lines.append(f"  Video B duration: {_fmt_dur(dur_b)}")
    if dur_a is not None and dur_b is not None:
        diff = abs(dur_a - dur_b)
        longer = "A" if dur_a > dur_b else ("B" if dur_b > dur_a else "same")
        lines.append(f"  Duration difference: {_fmt_dur(diff)} (Video {longer} is longer)")


def _append_date_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    date_a = _parse_date(video_a.metadata.upload_date)
    date_b = _parse_date(video_b.metadata.upload_date)
    lines.append("\n[Upload Dates]")
    lines.append(f"  Video A upload date: {video_a.metadata.upload_date or 'unknown'}")
    lines.append(f"  Video B upload date: {video_b.metadata.upload_date or 'unknown'}")
    if date_a and date_b:
        diff_days = abs((date_a - date_b).days)
        newer = "A" if date_a > date_b else ("B" if date_b > date_a else "same day")
        lines.append(f"  Days apart: {diff_days} (Video {newer} is newer)")
    else:
        lines.append("  Date comparison: unavailable")


def _append_follower_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    followers_a = video_a.metadata.follower_count
    followers_b = video_b.metadata.follower_count
    lines.append("\n[Follower / Subscriber Counts]")
    lines.append(f"  Video A creator followers: {_fmt_int(followers_a)}")
    lines.append(f"  Video B creator followers: {_fmt_int(followers_b)}")
    if followers_a is not None and followers_b is not None:
        diff = abs(followers_a - followers_b)
        bigger = "A" if followers_a > followers_b else ("B" if followers_b > followers_a else "equal")
        lines.append(f"  Follower count difference: {diff:,} (Video {bigger} creator has more)")


def _append_hashtag_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    tags_a = {t.lower().lstrip("#") for t in (video_a.metadata.hashtags or [])}
    tags_b = {t.lower().lstrip("#") for t in (video_b.metadata.hashtags or [])}
    overlap = tags_a & tags_b
    only_a = tags_a - tags_b
    only_b = tags_b - tags_a
    lines.append("\n[Hashtags]")
    lines.append(f"  Video A hashtags: {_fmt_tags(video_a.metadata.hashtags)}")
    lines.append(f"  Video B hashtags: {_fmt_tags(video_b.metadata.hashtags)}")
    lines.append(f"  Shared hashtags: {', '.join(sorted(overlap)) or 'none'}")
    lines.append(f"  Unique to A: {', '.join(sorted(only_a)) or 'none'}")
    lines.append(f"  Unique to B: {', '.join(sorted(only_b)) or 'none'}")


def _append_title_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    title_a = video_a.metadata.title or ""
    title_b = video_b.metadata.title or ""
    lines.append("\n[Titles]")
    lines.append(f"  Video A title: {title_a or 'unknown'}")
    lines.append(f"  Video B title: {title_b or 'unknown'}")
    if title_a and title_b:
        words_a = set(re.findall(r"[a-z]+", title_a.lower()))
        words_b = set(re.findall(r"[a-z]+", title_b.lower()))
        shared = words_a & words_b - {"the", "a", "an", "in", "of", "and", "or", "to", "for", "is", "are"}
        note = f"shared topic words: {', '.join(sorted(shared))}" if shared else "no notable word overlap"
        lines.append(f"  Title similarity note: {note}")


def _append_cta_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    cta_a = _detect_cta(video_a.transcript)
    cta_b = _detect_cta(video_b.transcript)
    lines.append("\n[Call-to-Action Signals]")
    lines.append(f"  Video A CTA detected: {'yes — ' + ', '.join(sorted(cta_a)) if cta_a else 'no'}")
    lines.append(f"  Video B CTA detected: {'yes — ' + ', '.join(sorted(cta_b)) if cta_b else 'no'}")


def _append_pacing_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    pace_a = _avg_words_per_segment(video_a.transcript)
    pace_b = _avg_words_per_segment(video_b.transcript)
    lines.append("\n[Transcript Pacing (avg words/segment)]")
    lines.append(f"  Video A: {_fmt_float(pace_a)}")
    lines.append(f"  Video B: {_fmt_float(pace_b)}")
    if pace_a is not None and pace_b is not None:
        note = "Video A is denser" if pace_a > pace_b else ("Video B is denser" if pace_b > pace_a else "similar pacing")
        lines.append(f"  Pacing note: {note}")


def _append_energy_section(
    lines: List[str],
    video_a: VideoAnalysis,
    video_b: VideoAnalysis,
) -> None:
    eng_a = _transcript_energy(video_a.transcript)
    eng_b = _transcript_energy(video_b.transcript)
    lines.append("\n[Transcript Energy (exclamation sentence ratio)]")
    lines.append(f"  Video A: {_fmt_pct(eng_a)}")
    lines.append(f"  Video B: {_fmt_pct(eng_b)}")
    lines.append(
        "  Note: higher ratio may indicate more enthusiastic or energetic delivery."
    )


def _rate(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if not denominator or numerator is None:
        return None
    return round(numerator / denominator * 100, 4)


def _detect_cta(transcript: List[TranscriptSegment]) -> Set[str]:
    combined = " ".join(s.text.lower() for s in transcript)
    return {word for word in _CTA_WORDS if word in combined}


def _avg_words_per_segment(transcript: List[TranscriptSegment]) -> Optional[float]:
    if not transcript:
        return None
    total = sum(len(s.text.split()) for s in transcript)
    return round(total / len(transcript), 2)


def _transcript_energy(transcript: List[TranscriptSegment]) -> Optional[float]:
    """Ratio of sentences ending in '!' to total sentences."""
    sentences = []
    for seg in transcript:
        sentences.extend(re.split(r"[.!?]", seg.text))
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return None
    exclaim = sum(1 for s in transcript if s.text.strip().endswith("!"))
    return round(exclaim / max(len(transcript), 1) * 100, 2)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    # yt-dlp returns YYYYMMDD format
    try:
        return datetime.strptime(value.strip(), "%Y%m%d")
    except ValueError:
        pass
    # ISO format fallback
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value:.4f}%" if value is not None else "unavailable"


def _fmt_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "unavailable"


def _fmt_float(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "unavailable"


def _fmt_dur(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    total = float(value)
    mm = int(total // 60)
    ss = total % 60
    seconds_text = f"{ss:06.3f}".rstrip("0").rstrip(".")
    return f"{mm}:{seconds_text.zfill(2)} ({total:.3f}s)"


def _fmt_tags(tags: Optional[List[str]]) -> str:
    if not tags:
        return "none"
    return ", ".join(tags[:15])
