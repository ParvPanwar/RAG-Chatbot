from collections.abc import AsyncIterator
import json
import re
from typing import Optional
from uuid import uuid4

import httpx
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.schemas.chat import ChatResponse, Citation
from app.services.analysis_store import get_analysis_result
from app.services.context_builder import build_context
from app.services.query_router import classify_query

_conversation_memory: dict[str, ConversationBufferWindowMemory] = {}


class ChatConfigurationError(ValueError):
    pass


def answer_question(
    analysis_id: str,
    message: str,
    conversation_id: Optional[str] = None,
) -> ChatResponse:
    conversation_id = conversation_id or str(uuid4())
    if not looks_like_video_question(message):
        answer = off_topic_answer()
        remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            citations=[],
        )

    analysis = get_analysis_result(analysis_id)
    intent = classify_query(message)
    context_text, docs = build_context(intent, analysis_id, message, analysis)

    if not docs and intent == "general_rag":
        answer = (
            "I do not have enough indexed transcript context to answer that question. "
            "Run analysis first, then ask about content present in the retrieved chunks."
        )
        remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            citations=[],
        )

    messages = build_chat_messages(conversation_id, message, context_text, intent)
    answer = ask_ollama_once(messages)

    remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        citations=build_citations(docs),
    )


async def stream_answer(
    analysis_id: str,
    message: str,
    conversation_id: Optional[str] = None,
) -> AsyncIterator[str]:
    conversation_id = conversation_id or str(uuid4())
    if not looks_like_video_question(message):
        answer = off_topic_answer()
        remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
        yield make_sse_event("conversation", conversation_id)
        yield make_sse_event("token", answer)
        yield make_sse_event("citations", [])
        yield make_sse_event("done", "[DONE]")
        return

    analysis = get_analysis_result(analysis_id)
    intent = classify_query(message)
    context_text, docs = build_context(intent, analysis_id, message, analysis)

    if not docs and intent == "general_rag":
        answer = (
            "I do not have enough indexed transcript context to answer that question. "
            "Run analysis first, then ask about content present in the retrieved chunks."
        )
        remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
        yield make_sse_event("conversation", conversation_id)
        yield make_sse_event("token", answer)
        yield make_sse_event("citations", [])
        yield make_sse_event("done", "[DONE]")
        return

    messages = build_chat_messages(conversation_id, message, context_text, intent)
    yield make_sse_event("conversation", conversation_id)

    answer_parts: list[str] = []
    async for token in stream_ollama_tokens(messages):
        if token:
            answer_parts.append(token)
            yield make_sse_event("token", token)

    answer = "".join(answer_parts)
    remember(conversation_id, HumanMessage(content=message), AIMessage(content=answer))
    yield make_sse_event("citations", [citation.model_dump() for citation in build_citations(docs)])
    yield make_sse_event("done", "[DONE]")


def build_chat_messages(
    conversation_id: str,
    message: str,
    context_text: str,
    intent: str,
) -> list[BaseMessage]:
    system_prompt = _build_system_prompt(intent)
    return [
        SystemMessage(content=system_prompt),
        *get_memory_messages(conversation_id),
        HumanMessage(
            content=(
                f"Context:\n{context_text}\n\n"
                f"Question:\n{message}"
            )
        ),
    ]


def _build_system_prompt(intent: str) -> str:
    """
    Return a system prompt tuned to the current question intent.

    All variants share a core instruction set:
    - Use computed metrics before guessing.
    - Distinguish facts (cite) from inferences (hedge).
    - Use phrases like "likely", "may indicate", or "based on available data".
    - Say when evidence is unavailable instead of fabricating values.
    """
    core = (
        "You are a social media video comparison assistant. "
        "Two videos have been analysed: Video A (first) and Video B (second). "
        "Always use the provided data rather than general knowledge. "
        "\n\n"
        "FACTUAL CLAIMS: Every claim about specific numbers, transcript content, "
        "or metadata must be supported by the provided context. "
        "Cite transcript-based claims using labels like [Video A - Chunk A-3] "
        "or [Video A - Chunk A-3, 00:05-00:18] when a timestamp range is given. "
        "\n\n"
        "INFERENCES: When drawing conclusions that go beyond the raw data, "
        "use hedging language: 'likely', 'may indicate', 'based on available data', "
        "'this could suggest', or 'it appears'. "
        "Never claim causation with certainty from engagement data alone. "
        "\n\n"
        "MISSING DATA: If specific metadata or transcript evidence is not in the "
        "context, say so explicitly. Do not fabricate values. "
        "\n\n"
        "SCOPE: Only answer questions about the analysed videos, their metadata, "
        "transcripts, or comparison. Politely decline off-topic questions."
    )

    intent_addenda = {
        "metadata_question": (
            "\n\nThis is a METADATA question. "
            "The context contains exact computed metrics. "
            "Lead with the precise figures before any interpretation. "
            "Do not retrieve or guess additional values."
        ),
        "hook_question": (
            "\n\nThis is a HOOK / OPENING question. "
            "The context includes timestamped opening transcript windows (first 5s, 15s, 30s). "
            "Compare the opening lines of both videos directly, quoting specific phrases. "
            "Note which video establishes its value proposition faster."
        ),
        "performance_reasoning": (
            "\n\nThis is a PERFORMANCE REASONING question. "
            "Start with the computed metrics provided. "
            "Distinguish between what the data shows (fact) and why it might have happened (inference). "
            "Consider multiple factors: hook strength, CTA presence, pacing, follower advantage, and content timing. "
            "Avoid claiming a single root cause."
        ),
        "improvement_question": (
            "\n\nThis is an IMPROVEMENT / SUGGESTION question. "
            "Base suggestions on concrete differences observed in the metrics and transcript. "
            "Reference what the stronger-performing video does that the other does not. "
            "Frame suggestions as possibilities, not guarantees."
        ),
    }

    return core + intent_addenda.get(intent, "")


def get_ollama_settings() -> tuple[str, str, str]:
    settings = get_settings()
    if not settings.ollama_base_url:
        raise ChatConfigurationError(
            "OLLAMA_BASE_URL is not set. Add your remote Ollama server URL to your .env file."
        )
    return (
        settings.ollama_base_url.rstrip("/"),
        settings.ollama_chat_model,
        settings.ollama_api_key,
    )


def ask_ollama_once(messages: list[BaseMessage]) -> str:
    base_url, model, api_key = get_ollama_settings()
    response = httpx.post(
        f"{base_url}/api/chat",
        headers=ollama_headers(api_key),
        json=build_ollama_payload(model, messages, stream=False),
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(extract_ollama_error(response))
    return str(response.json().get("message", {}).get("content", ""))


async def stream_ollama_tokens(messages: list[BaseMessage]) -> AsyncIterator[str]:
    base_url, model, api_key = get_ollama_settings()
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat",
            headers=ollama_headers(api_key),
            json=build_ollama_payload(model, messages, stream=True),
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(extract_ollama_error_from_body(body))
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield str(payload.get("message", {}).get("content", ""))


def build_ollama_payload(model: str, messages: list[BaseMessage], stream: bool) -> dict:
    return {
        "model": model,
        "messages": [to_ollama_message(msg) for msg in messages],
        "stream": stream,
        "options": {"temperature": 0.2},
    }


def to_ollama_message(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "user"
    return {"role": role, "content": str(message.content)}


def ollama_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_ollama_error(response: httpx.Response) -> str:
    return extract_ollama_error_from_body(response.content)


def extract_ollama_error_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
        return payload.get("error") or payload.get("message") or str(payload)
    except Exception:
        return body.decode("utf-8", errors="replace")


def friendly_chat_error(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, ChatConfigurationError):
        return message

    connection_markers = ("connection refused", "name or service not known", "nodename", "timed out")
    if any(marker.lower() in message.lower() for marker in connection_markers):
        return "Could not reach the remote Ollama server. Check OLLAMA_BASE_URL and network access."

    auth_markers = ("unauthorized", "forbidden", "401", "403")
    if any(marker.lower() in message.lower() for marker in auth_markers):
        return "Remote Ollama authentication failed. Check OLLAMA_API_KEY or your server access settings."

    return f"Chat generation failed: {message}"


def looks_like_video_question(message: str) -> bool:
    """Cheap guard so random text does not spend LLM tokens."""
    text = message.strip().lower()
    if len(text) < 4:
        return False

    tokens = re.findall(r"[a-z0-9']+", text)
    if not tokens:
        return False

    if len(tokens) == 1 and not tokens[0].isdigit():
        return tokens[0] in {
            "summary", "summarize", "compare", "engagement", "views",
            "likes", "comments", "duration", "transcript", "creator", "title",
        }

    letters = re.findall(r"[a-z]", text)
    vowels = re.findall(r"[aeiou]", text)
    if letters and len(vowels) / len(letters) < 0.18:
        return False

    signal_words = {
        "what", "which", "who", "why", "how", "when", "where",
        "compare", "comparison", "summarize", "summary", "explain",
        "video", "engagement", "views", "likes", "comments", "duration",
        "creator", "title", "transcript", "hook", "topic", "performance",
        "better", "higher", "lower", "difference", "similar", "audience",
        "content", "improve", "suggest", "reason", "opening", "intro", "a", "b",
    }
    return any(token in signal_words for token in tokens) or "?" in message


def off_topic_answer() -> str:
    return (
        "Please ask a clear question about the analysed videos, their transcript, "
        "metadata, or engagement comparison."
    )


def remember(conversation_id: str, user_message: HumanMessage, ai_message: AIMessage) -> None:
    memory = get_conversation_memory(conversation_id)
    memory.chat_memory.add_message(user_message)
    memory.chat_memory.add_message(ai_message)


def get_memory_messages(conversation_id: str) -> list[BaseMessage]:
    memory = get_conversation_memory(conversation_id)
    return list(memory.load_memory_variables({}).get("history", []))


def get_conversation_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    if conversation_id not in _conversation_memory:
        _conversation_memory[conversation_id] = ConversationBufferWindowMemory(
            k=6,
            return_messages=True,
            memory_key="history",
        )
    return _conversation_memory[conversation_id]


def build_citations(documents) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []

    for doc in documents:
        key = str(doc.metadata.get("chunk_id", ""))
        if key in seen:
            continue
        seen.add(key)

        start_time = doc.metadata.get("start_time")
        end_time = doc.metadata.get("end_time")

        citations.append(
            Citation(
                citation=build_citation_label(doc),
                video_id=str(doc.metadata.get("video_id", "")),
                title=doc.metadata.get("title") or None,
                creator=doc.metadata.get("creator") or None,
                chunk_id=key,
                chunk_text=doc.page_content,
                start_time=float(start_time) if start_time is not None else None,
                end_time=float(end_time) if end_time is not None else None,
            )
        )

    return citations


def build_citation_label(document) -> str:
    """
    Build a citation label, including timestamp range when available.
    e.g.  [Video A - Chunk A-3, 00:05-00:18]
          [Video A - Chunk A-3]   ← fallback when no timing
    """
    metadata = document.metadata
    video_letter = str(metadata.get("video_label") or "A").upper()
    chunk_index = int(metadata.get("chunk_index", 0)) + 1
    base = f"Video {video_letter} - Chunk {video_letter}-{chunk_index}"

    start = metadata.get("start_time")
    end = metadata.get("end_time")
    if start is not None and end is not None:
        return f"{base}, {_fmt_ts(float(start))}-{_fmt_ts(float(end))}"
    return base


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def make_sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
