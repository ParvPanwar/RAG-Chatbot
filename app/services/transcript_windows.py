from typing import List

from app.schemas.common import TranscriptSegment


def get_window(segments: List[TranscriptSegment], max_seconds: float) -> List[TranscriptSegment]:
    """Return segments whose start time is strictly less than *max_seconds*."""
    return [s for s in segments if s.start < max_seconds]


def get_opening_5s(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    return get_window(segments, 5.0)


def get_opening_15s(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    return get_window(segments, 15.0)


def get_opening_30s(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    return get_window(segments, 30.0)


def format_window_text(segments: List[TranscriptSegment]) -> str:
    """
    Render segments as a timestamped transcript block, e.g.:
        [0.00s] Hello everyone…
        [3.21s] Today we're comparing…
    """
    if not segments:
        return "(no transcript content in this window)"
    return "\n".join(f"[{s.start:.2f}s] {s.text}" for s in segments if s.text.strip())


def format_seconds(seconds: float) -> str:
    """Convert fractional seconds to MM:SS string for display."""
    total = int(seconds)
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"
