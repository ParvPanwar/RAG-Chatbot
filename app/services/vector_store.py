import logging
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.schemas.common import VideoAnalysis

logger = logging.getLogger(__name__)

# Reuse the Chroma client so repeated requests do not reopen the DB.
_vector_store = None


class VectorStoreError(ValueError):
    pass


def index_video_transcripts(analysis_id: str, videos: list[VideoAnalysis]) -> int:
    try:
        docs = build_documents(analysis_id, videos)
        if not docs:
            logger.info(f"No transcripts to index for analysis_id {analysis_id}")
            return 0

        logger.info(f"Indexing {len(docs)} transcript chunks for analysis_id {analysis_id}")
        store = get_vector_store()
        store.add_documents(
            docs,
            ids=[str(doc.metadata["chunk_id"]) for doc in docs],
        )
        logger.info(f"Successfully indexed {len(docs)} chunks for analysis_id {analysis_id}")
        return len(docs)
    except Exception as err:
        logger.error(f"Error indexing transcripts for analysis_id {analysis_id}: {err}", exc_info=True)
        raise VectorStoreError(f"Failed to index transcripts in vector store: {err}") from err


def search_transcripts(
    query: str,
    analysis_id: str = None,
    top_k: int = None,
) -> list[Document]:
    try:
        settings = get_settings()
        store = get_vector_store()
        search_opts = {"k": top_k or settings.rag_top_k}
        if analysis_id:
            search_opts["filter"] = {"analysis_id": analysis_id}

        logger.info(f"Searching transcripts for query: '{query}' (analysis_id: {analysis_id}, top_k: {search_opts['k']})")
        matches = store.similarity_search(query, **search_opts)
        logger.info(f"Found {len(matches)} relevant chunks in vector store")
        return matches
    except Exception as err:
        logger.error(f"Error searching transcripts: {err}", exc_info=True)
        raise VectorStoreError(f"Failed to retrieve context from vector store: {err}") from err


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        try:
            settings = get_settings()
            logger.info(f"Initializing Chroma vector store with local embeddings ({settings.local_embedding_model})...")
            from langchain_huggingface import HuggingFaceEmbeddings
            embedder = HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            _vector_store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embedder,
                persist_directory=settings.chroma_persist_dir,
            )
            logger.info(f"Chroma DB successfully connected to {settings.chroma_persist_dir} with local embeddings")
        except Exception as err:
            logger.error(f"Failed to initialize Chroma vector store: {err}", exc_info=True)
            if isinstance(err, VectorStoreError):
                raise err
            raise VectorStoreError(f"Failed to initialize Chroma vector store: {err}") from err
    return _vector_store


def build_documents(analysis_id: str, videos: list[VideoAnalysis]) -> list[Document]:
    """
    Build LangChain Documents from transcript segments.

    Segments are accumulated into chunks by character count (≈900 chars with
    ~120-char overlap window) so each chunk retains an exact start_time and
    end_time derived from its constituent segments.  This enables
    timestamp-aware citations like [Video A - Chunk A-3, 00:05-00:18].
    """
    settings = get_settings()
    docs: list[Document] = []

    for video_position, video in enumerate(videos):
        video_label = video.video_label or ("A" if video_position == 0 else "B")
        segments = [s for s in video.transcript if s.text.strip()]
        if not segments:
            continue

        timed_chunks = _chunk_segments(segments, chunk_size=900, chunk_overlap=120)
        timed_chunks = timed_chunks[: settings.max_chunks_per_video]

        for chunk_index, (chunk_text, start_time, end_time) in enumerate(timed_chunks):
            chunk_id = f"{analysis_id}:{video_label}:{video.platform}:{video.video_id}:{chunk_index}"
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "analysis_id": analysis_id,
                        "platform": video.platform,
                        "video_id": video.video_id,
                        "video_label": video_label,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        # Used later for timestamped citations in chat.
                        "start_time": start_time,
                        "end_time": end_time,
                        "title": video.metadata.title or "",
                        "creator": video.metadata.creator or "",
                        "follower_count": video.metadata.follower_count or 0,
                        "hashtags": ", ".join(video.metadata.hashtags),
                        "source_url": video.url,
                    },
                )
            )

    return docs


def _chunk_segments(
    segments: list,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[Tuple[str, float, float]]:
    """
    Group consecutive TranscriptSegments into text chunks.

    Returns list of (chunk_text, start_time_secs, end_time_secs).
    The overlap is implemented by back-tracking into the previous chunk's
    tail segments when starting a new chunk.
    """
    chunks: List[Tuple[str, float, float]] = []
    open_chunk: list = []
    open_len: int = 0

    for seg in segments:
        line = f"[{seg.start:.2f}s] {seg.text}"
        line_len = len(line)

        if open_len + line_len > chunk_size and open_chunk:
            chunks.append(_make_chunk(open_chunk))

            # Keep the tail so adjacent chunks have a little shared context.
            overlap_segs: list = []
            overlap_len = 0
            for prev_seg in reversed(open_chunk):
                prev_line = f"[{prev_seg.start:.2f}s] {prev_seg.text}"
                if overlap_len + len(prev_line) > chunk_overlap:
                    break
                overlap_segs.insert(0, prev_seg)
                overlap_len += len(prev_line)

            open_chunk = overlap_segs
            open_len = overlap_len

        open_chunk.append(seg)
        open_len += line_len

    if open_chunk:
        chunks.append(_make_chunk(open_chunk))

    return chunks


def _make_chunk(segs: list) -> Tuple[str, float, float]:
    """Render a list of segments into (text, start_time, end_time)."""
    text = "\n".join(f"[{s.start:.2f}s] {s.text}" for s in segs)
    start_time = segs[0].start
    # end_time = start of last segment + its duration; fall back to its start
    last = segs[-1]
    end_time = last.start + (last.duration or 0.0)
    return text, start_time, end_time
