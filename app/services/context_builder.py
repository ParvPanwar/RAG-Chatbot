from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.schemas.analysis import AnalyzeResponse
from app.services.analysis_store import get_analysis_result
from app.services.performance_metrics import format_performance_metrics
from app.services.transcript_windows import (
    format_window_text,
    get_opening_15s,
    get_opening_30s,
    get_opening_5s,
)
from app.services.vector_store import search_transcripts

logger = logging.getLogger(__name__)

ContextResult = Tuple[str, List[Document]]


def build_context(
    intent: str,
    analysis_id: str,
    message: str,
    analysis: Optional[AnalyzeResponse],
) -> ContextResult:
    """
    Route to the appropriate context builder based on *intent*.

    Returns ``(context_text, documents_for_citations)``.
    """
    if analysis is None:
        return "No analysis data is available.", []

    builders = {
        "metadata_question":   _build_metadata_context,
        "hook_question":       _build_hook_context,
        "performance_reasoning": _build_performance_context,
        "improvement_question": _build_improvement_context,
        "general_rag":         _build_general_context,
    }
    builder = builders.get(intent, _build_general_context)
    return builder(analysis_id, message, analysis)


def _build_metadata_context(
    analysis_id: str,
    message: str,
    analysis: AnalyzeResponse,
) -> ContextResult:
    """
    Metadata questions: use only exact structured data.
    No vector search — avoids irrelevant transcript chunks polluting the answer.
    """
    summary = _format_both_videos(analysis)
    metrics = format_performance_metrics(analysis.video_a, analysis.video_b)

    context = (
        "=== Video Metadata (exact values) ===\n"
        f"{summary}\n\n"
        f"{metrics}\n\n"
        "Answer using ONLY the figures above. Do not retrieve or guess "
        "additional values from the transcript."
    )
    return context, []   # no vector docs needed


def _build_hook_context(
    analysis_id: str,
    message: str,
    analysis: AnalyzeResponse,
) -> ContextResult:
    """
    Hook questions: metadata + perf metrics + opening windows + top-3 vector chunks.
    """
    summary = _format_both_videos(analysis)
    metrics = format_performance_metrics(analysis.video_a, analysis.video_b)
    windows = _format_opening_windows(analysis, seconds=[5, 15, 30])

    docs = search_transcripts(query=message, analysis_id=analysis_id, top_k=3)
    chunk_block = _format_documents(docs)

    context = (
        "=== Video Metadata ===\n"
        f"{summary}\n\n"
        f"{metrics}\n\n"
        "=== Opening Transcript Windows ===\n"
        f"{windows}\n\n"
        "=== Additional Relevant Transcript Chunks ===\n"
        f"{chunk_block}"
    )
    return context, docs


def _build_performance_context(
    analysis_id: str,
    message: str,
    analysis: AnalyzeResponse,
) -> ContextResult:
    """
    Performance reasoning: full metadata + metrics + opening 15s/30s + top-6 vector chunks.
    """
    summary = _format_both_videos(analysis)
    metrics = format_performance_metrics(analysis.video_a, analysis.video_b)
    windows = _format_opening_windows(analysis, seconds=[15, 30])

    docs = search_transcripts(query=message, analysis_id=analysis_id, top_k=6)
    chunk_block = _format_documents(docs)

    context = (
        "=== Video Metadata ===\n"
        f"{summary}\n\n"
        f"{metrics}\n\n"
        "=== Opening Transcript Windows (15s and 30s) ===\n"
        f"{windows}\n\n"
        "=== Relevant Transcript Chunks ===\n"
        f"{chunk_block}"
    )
    return context, docs


def _build_improvement_context(
    analysis_id: str,
    message: str,
    analysis: AnalyzeResponse,
) -> ContextResult:
    """
    Improvement suggestions: metadata + metrics + opening 30s + top-6 vector chunks.
    """
    summary = _format_both_videos(analysis)
    metrics = format_performance_metrics(analysis.video_a, analysis.video_b)
    windows = _format_opening_windows(analysis, seconds=[30])

    docs = search_transcripts(query=message, analysis_id=analysis_id, top_k=6)
    chunk_block = _format_documents(docs)

    context = (
        "=== Video Metadata ===\n"
        f"{summary}\n\n"
        f"{metrics}\n\n"
        "=== Opening Transcript Windows (first 30s) ===\n"
        f"{windows}\n\n"
        "=== Relevant Transcript Chunks ===\n"
        f"{chunk_block}"
    )
    return context, docs


def _build_general_context(
    analysis_id: str,
    message: str,
    analysis: AnalyzeResponse,
) -> ContextResult:
    """
    General RAG: keep existing vector-only behaviour, but now also include
    a short metadata header so the LLM always has basic facts.
    """
    summary = _format_both_videos(analysis)
    docs = search_transcripts(query=message, analysis_id=analysis_id, top_k=6)
    chunk_block = _format_documents(docs)

    context = (
        "=== Video Metadata ===\n"
        f"{summary}\n\n"
        "=== Retrieved Transcript Chunks ===\n"
        f"{chunk_block}"
    )
    return context, docs


def _format_both_videos(analysis: AnalyzeResponse) -> str:
    lines = [
        _format_video_block("A", analysis.video_a),
        _format_video_block("B", analysis.video_b),
    ]
    if analysis.comparison:
        c = analysis.comparison
        lines.append(
            f"\nComparison summary: winner={c.higher_engagement_platform}, "
            f"engagement_rate_gap={c.engagement_rate_gap}, "
            f"duration_difference_seconds={c.duration_difference_seconds}"
        )
    return "\n".join(lines)


def _format_video_block(label: str, video) -> str:
    m = video.metadata
    return (
        f"Video {label}: platform={video.platform}, video_id={video.video_id}, "
        f"title={m.title or 'unknown'}, creator={m.creator or 'unknown'}, "
        f"views={m.views}, likes={m.likes}, comments={m.comments}, "
        f"duration_seconds={m.duration}, "
        f"engagement_rate={video.engagement_rate}, "
        f"upload_date={m.upload_date or 'unknown'}, "
        f"follower_count={m.follower_count}, "
        f"hashtags={', '.join(m.hashtags) if m.hashtags else 'none'}"
    )


def _format_opening_windows(analysis: AnalyzeResponse, seconds: List[int]) -> str:
    """
    Build a labelled block of opening transcript windows for all requested
    second thresholds for both videos.
    """
    video_map = [
        ("A", analysis.video_a),
        ("B", analysis.video_b),
    ]
    window_fns = {
        5:  get_opening_5s,
        15: get_opening_15s,
        30: get_opening_30s,
    }

    blocks = []
    for label, video in video_map:
        for sec in seconds:
            window_getter = window_fns.get(sec)
            if window_getter is None:
                continue
            window = window_getter(video.transcript)
            text = format_window_text(window)
            blocks.append(f"[Video {label} — first {sec}s]\n{text}")
    return "\n\n".join(blocks) if blocks else "(no transcript windows available)"


def _format_documents(docs: List[Document]) -> str:
    """Format retrieved LangChain documents into a readable context block."""
    if not docs:
        return "(no relevant transcript chunks retrieved)"
    return "\n\n".join(_format_one_doc(doc) for doc in docs)


def _format_one_doc(doc: Document) -> str:
    meta = doc.metadata
    label = _build_citation_label(doc)
    title = meta.get("title") or "Unknown title"
    creator = meta.get("creator") or "Unknown creator"
    video_id = meta.get("video_id") or "unknown"

    # Include timing when the chunk was built from timestamped segments.
    start = meta.get("start_time")
    end = meta.get("end_time")
    timing_str = ""
    if start is not None and end is not None:
        timing_str = f" [{_fmt_ts(start)}–{_fmt_ts(end)}]"

    return (
        f"[{label}{timing_str}]\n"
        f"Video ID: {video_id}\n"
        f"Title: {title}\n"
        f"Creator: {creator}\n"
        f"Transcript:\n{doc.page_content}"
    )


def _build_citation_label(doc: Document) -> str:
    meta = doc.metadata
    letter = str(meta.get("video_label") or "A").upper()
    chunk_no = int(meta.get("chunk_index", 0)) + 1
    return f"Video {letter} - Chunk {letter}-{chunk_no}"


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"
