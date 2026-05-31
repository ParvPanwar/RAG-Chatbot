from fastapi import APIRouter, HTTPException, status

from app.schemas.retrieval import RetrieveRequest, RetrieveResponse
from app.services.retriever import retrieve_relevant_chunks
from app.services.vector_store import VectorStoreError

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        return retrieve_relevant_chunks(
            query=request.query,
            analysis_id=request.analysis_id,
            top_k=request.top_k,
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
