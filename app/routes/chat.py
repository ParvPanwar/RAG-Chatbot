import json
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_chat import answer_question, friendly_chat_error, stream_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(f"POST /chat request received: analysis_id={request.analysis_id}, conversation_id={request.conversation_id}, message='{request.message}'")
    try:
        response = answer_question(
            analysis_id=request.analysis_id,
            message=request.message,
            conversation_id=request.conversation_id,
        )
        logger.info(f"POST /chat success: conversation_id={response.conversation_id}")
        return response
    except Exception as exc:
        logger.error(f"POST /chat failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=friendly_chat_error(exc),
        ) from exc


@router.post("/stream")
def stream_chat(request: ChatRequest) -> StreamingResponse:
    logger.info(f"POST /chat/stream request received: analysis_id={request.analysis_id}, conversation_id={request.conversation_id}, message='{request.message}'")

    async def events():
        try:
            async for event in stream_answer(
                analysis_id=request.analysis_id,
                message=request.message,
                conversation_id=request.conversation_id,
            ):
                yield event
        except Exception as exc:
            logger.error(f"Error in chat SSE stream: {exc}", exc_info=True)
            yield f"event: error\ndata: {json.dumps(friendly_chat_error(exc))}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
