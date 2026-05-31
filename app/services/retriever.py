from app.schemas.retrieval import RetrievedChunk, RetrieveResponse
from app.services.vector_store import search_transcripts


def retrieve_relevant_chunks(
    query: str,
    analysis_id: str = None,
    top_k: int = 5,
) -> RetrieveResponse:
    documents = search_transcripts(
        query=query,
        analysis_id=analysis_id,
        top_k=top_k,
    )

    return RetrieveResponse(
        chunks=[
            RetrievedChunk(
                chunk_text=document.page_content,
                chunk_id=str(document.metadata.get("chunk_id", "")),
                video_id=str(document.metadata.get("video_id", "")),
            )
            for document in documents
        ]
    )
