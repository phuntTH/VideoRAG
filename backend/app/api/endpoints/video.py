import os
import json
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.config import settings
from app.core.jobs import job_store
from app.models.request import VideoProcessRequest
from app.models.response import (
    VideoProcessResponse,
    VideoStatusResponse,
    VideoTranscriptResponse,
    VideoSuggestionsResponse
)
from app.services.transcript.source_selector import TranscriptSourceSelector
from app.services.preprocessing.chunking import SemanticChunker
from app.services.vector_store.faiss_store import FAISSStore
from app.services.retrieval.sparse_search import BM25Store

logger = logging.getLogger("video_rag_backend")
router = APIRouter()

# Define and ensure core storage directories exist dynamically
STORAGE_DIR = "storage"
TRANSCRIPTS_DIR = os.path.join(STORAGE_DIR, "transcripts")
SUGGESTIONS_DIR = os.path.join(STORAGE_DIR, "suggestions")

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(SUGGESTIONS_DIR, exist_ok=True)


def run_ingestion_pipeline(url_str: str, video_id: str):
    """
    Executes the comprehensive machine learning pipeline on a background thread.
    Safely synchronizes data keys across decoupled processing services.
    """
    try:
        job_store.update_status(video_id, "downloading")
        selector = TranscriptSourceSelector()
        raw_transcript = selector.get_raw_transcript(url_str, video_id)
        if not raw_transcript:
            raise ValueError("Unable to retrieve transcript data from the video asset.")

        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(raw_transcript, f, ensure_ascii=False, indent=4)
        formatted_sentences = [
            {
                "text": item["text"],
                "start_time": item["start"],
                "end_time": item["start"] + item["duration"],
                "video_id": video_id
            }
            for item in raw_transcript
        ]

        job_store.update_status(video_id, "chunking")
        chunker = SemanticChunker(percentile_threshold=90.0)
        chunks = chunker.chunk_sentences(formatted_sentences)
        if not chunks:
            raise ValueError("Text segmentation resulted in an empty output sequence.")

        job_store.update_status(video_id, "embedding")
        dense_store = FAISSStore()
        dense_store.build_index(chunks=chunks, video_id=video_id)
        job_store.update_status(video_id, "indexing")
        sparse_store = BM25Store()
        sparse_store.build_index(chunks)
        sparse_store.save_index(settings.BM25_INDICES_DIR, index_name=video_id)
        dynamic_suggestions = [
            "Tóm tắt nội dung chính được đề cập trong đoạn video này?",
            "Điểm cốt lõi hoặc giải pháp quan trọng nhất trong video là gì?",
            "Nêu các mốc thời gian hoặc ví dụ thực tế được người nói sử dụng?"
        ]

        if chunks and len(chunks) > 0:
            first_chunk_text = chunks[0].get("text", "")
            if first_chunk_text:
                short_text = first_chunk_text[:60].strip().replace("\n", " ")
                dynamic_suggestions.insert(0, f"Video nhắc gì về việc: '{short_text}...'?")
                dynamic_suggestions = dynamic_suggestions[:3]

        suggestions_path = os.path.join(SUGGESTIONS_DIR, f"{video_id}.json")
        with open(suggestions_path, "w", encoding="utf-8") as f:
            json.dump(dynamic_suggestions, f, ensure_ascii=False, indent=4)

        job_store.update_status(video_id, "completed")
        logger.info(f"🎉 Background ingestion pipeline completed successfully for video: {video_id}")

    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        job_store.update_status(video_id, "failed", error=error_msg)


# --- Endpoint Routers ---

@router.post("/process", response_model=VideoProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    url_str = str(request.url)
    selector = TranscriptSourceSelector()
    
    try:
        # Reuses the exact regular expression pattern from source_selector.py
        video_id = selector.extract_video_id(url_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    current_status = job_store.get_status(video_id)
    active_states = ["queued", "downloading", "chunking", "embedding", "indexing"]
    
    if current_status["status"] in active_states:
        return VideoProcessResponse(
            status=current_status["status"],
            message="Video ingestion pipeline is already actively running.",
            video_id=video_id
        )

    job_store.update_status(video_id, "queued")
    background_tasks.add_task(run_ingestion_pipeline, url_str, video_id)

    return VideoProcessResponse(
        status="queued",
        message="Video ingestion pipeline successfully triggered in the background.",
        video_id=video_id
    )


@router.get("/status", response_model=VideoStatusResponse)
async def get_pipeline_status(video_id: str):
    job_state = job_store.get_status(video_id)
    if job_state["status"] == "unknown":
        raise HTTPException(status_code=404, detail="No processing history found for the given video ID.")
        
    return VideoStatusResponse(
        video_id=video_id,
        status=job_state["status"],
        error=job_state["error"]
    )


@router.get("/transcript", response_model=VideoTranscriptResponse)
async def get_video_transcript(video_id: str):
    job_state = job_store.get_status(video_id)
    if job_state["status"] == "unknown":
        raise HTTPException(status_code=404, detail="No active or completed tasks matched this video ID.")
    if job_state["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Data is not ready yet. Target job state is: {job_state['status']}")

    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Target transcript asset file missing from local storage.")
        
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    formatted_transcript = [
        {
            "text": item.get("text", ""),
            "start": item.get("start", 0.0) if "start" in item else item.get("start_time", 0.0),
            "duration": item.get("duration", 0.0),
            "source_type": item.get("source_type", "youtube_caption")
        }
        for item in raw_data
    ]
        
    return VideoTranscriptResponse(video_id=video_id, transcript=formatted_transcript)


@router.get("/suggestions", response_model=VideoSuggestionsResponse)
async def get_video_suggestions(video_id: str):
    job_state = job_store.get_status(video_id)
    if job_state["status"] == "unknown":
        raise HTTPException(status_code=404, detail="No active or completed tasks matched this video ID.")
    if job_state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Sample questions cannot be requested until processing completes.")

    suggestions_path = os.path.join(SUGGESTIONS_DIR, f"{video_id}.json")
    if not os.path.exists(suggestions_path):
        raise HTTPException(status_code=404, detail="Target suggestions asset file missing from local storage.")
        
    with open(suggestions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return VideoSuggestionsResponse(video_id=video_id, suggestions=data)