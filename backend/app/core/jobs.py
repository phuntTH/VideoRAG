import threading
from typing import Dict, Optional, Any

class VideoJobStore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._jobs: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def update_status(self, video_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._jobs[video_id] = {
                "status": status,
                "error": error
            }

    def get_status(self, video_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._jobs.get(video_id, {"status": "unknown", "error": None})

job_store = VideoJobStore()