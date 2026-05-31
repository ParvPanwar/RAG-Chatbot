import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_chat import answer_question, friendly_chat_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(f"POST /chat request received: analysis_id={request.analysis_id}, message='{request.message}'")
    try:
        response = answer_question(
            analysis_id=request.analysis_id,
            message=request.message,
        )
        logger.info("POST /chat success")
        return response
    except Exception as exc:
        logger.error(f"POST /chat failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=friendly_chat_error(exc),
        ) from exc
