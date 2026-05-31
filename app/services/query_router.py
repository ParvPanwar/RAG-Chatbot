import re
from typing import Literal

QueryIntent = Literal[
    "hook_question",
    "improvement_question",
    "performance_reasoning",
    "metadata_question",
    "general_rag",
]

_HOOK_KEYWORDS = [
    "hook",
    "first 5",
    "first five",
    "first 15",
    "first fifteen",
    "first 30",
    "first thirty",
    "opening seconds",
    "opening line",
    "opening scene",
    "opening shot",
    "opening moment",
    "intro",
    "introduction",
    "start of",
    "beginning",
    "starts with",
    "opens with",
    "attention",
    "grab",
]

_IMPROVEMENT_KEYWORDS = [
    "improve",
    "improvement",
    "suggest",
    "suggestion",
    "recommendation",
    "recommend",
    "what worked",
    "what works",
    "tip",
    "tips",
    "better for",
    "how to make",
    "how could",
    "how should",
    "optimize",
    "optimise",
    "enhance",
    "what can",
    "do differently",
    "lessons",
    "take away",
    "takeaway",
    "learn from",
]

_PERFORMANCE_KEYWORDS = [
    "why did",
    "why does",
    "why is",
    "why has",
    "reason",
    "reasons",
    "explain why",
    "explain the",
    "what drove",
    "outperform",
    "out-perform",
    "perform better",
    "performing better",
    "performed better",
    "higher engagement",
    "more engagement",
    "more views",
    "more likes",
    "more comments",
    "lower engagement",
    "fewer views",
    "fewer likes",
    "what caused",
    "factor",
    "factors",
]

_METADATA_KEYWORDS = [
    "engagement rate",
    "engagement rates",
    "view count",
    "views",
    "like count",
    "likes",
    "comment count",
    "comments",
    "duration",
    "how long",
    "how many views",
    "how many likes",
    "how many comments",
    "follower",
    "followers",
    "subscriber",
    "subscribers",
    "upload date",
    "when was",
    "when were",
    "who made",
    "who created",
    "what's the",
    "what is the",
    "creator",
    "author",
    "hashtag",
    "hashtags",
    "tag",
    "tags",
    "platform",
]


def classify_query(message: str) -> QueryIntent:
    """
    Return the intent category for *message*.

    Priority order:
      hook_question > improvement_question > performance_reasoning
        > metadata_question > general_rag
    """
    text = message.lower()

    if _matches_any(text, _HOOK_KEYWORDS):
        return "hook_question"

    if _matches_any(text, _IMPROVEMENT_KEYWORDS):
        return "improvement_question"

    if _matches_any(text, _PERFORMANCE_KEYWORDS):
        return "performance_reasoning"

    if _matches_any(text, _METADATA_KEYWORDS):
        return "metadata_question"

    return "general_rag"


def _matches_any(text: str, keywords: list) -> bool:
    """True if *text* contains any keyword as a contiguous substring."""
    return any(kw in text for kw in keywords)
