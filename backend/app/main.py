import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.config import settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("video_rag_backend")

def ensure_infrastructure_storage():
    """
    Guarantees the existence of standard production disk partitions.
    Reduces critical pipeline failures caused by missing destination directories.
    """
    base_dirs = [
        "storage/audio_cache",
        "storage/faiss_indices",
        "storage/bm25_indices",
        "storage/response_cache"
    ]
    for target_dir in base_dirs:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            logger.info(f"Generated required base infrastructure directory: {target_dir}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the centralized application event loop boundaries (Startup and Teardown sequences).
    """
    logger.info("Initializing high-performance RAG backend startup validations...")
    ensure_infrastructure_storage()
    yield
    logger.info("De-allocating operational system resource contexts safely...")

app = FastAPI(
    title="Next-Gen Video Hybrid RAG Pipeline",
    description="Enterprise API backing dual dense-sparse vector space retrieval networks.",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Intercepts raw HTTP requests to log sub-resource latencies and 
    injects precise processing metrics directly inside response header frames.
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    execution_delta = time.time() - start_time
    
    response.headers["X-Process-Time"] = f"{execution_delta:.4f}s"
    
    logger.info(f"Trace Request: {request.method} {request.url.path} completed in {execution_delta:.4f} seconds.")
    
    return response

allowed_origins = getattr(settings, "ALLOWED_ORIGINS", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled background processing exceptions (Gemini timeouts, 
    missing FAISS indexes, serialization corruptions), hiding raw execution stack traces.
    """
    logger.exception(f"Unhandled operational exception captured during {request.method} targeting context path {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error encountered while executing inference loops."
        }
    )

@app.get("/api/health", tags=["System Diagnostics"])
async def system_health_check():
    """
    Examines active persistent mount paths to confirm infrastructure 
    stability and cache readiness without causing blocking IO operations.
    """
    storage_dependencies = [
        "storage/audio_cache",
        "storage/faiss_indices",
        "storage/bm25_indices"
    ]
    cache_dependencies = [
        "storage/response_cache"
    ]
    
    storage_ok = all(os.path.exists(path) for path in storage_dependencies)
    cache_ok = all(os.path.exists(path) for path in cache_dependencies)
    
    overall_status = "healthy" if (storage_ok and cache_ok) else "degraded"
    
    return {
        "status": overall_status,
        "storage_exists": storage_ok,
        "cache_exists": cache_ok,
        "epoch_timestamp": int(time.time())
    }

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)