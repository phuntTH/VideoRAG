import hashlib
import os
import pickle
from typing import Dict, Any, Optional

class PersistentSearchCache:
    def __init__(self, cache_dir: str = "./storage/response_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "cache.pkl")
        self._cache_store: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self._cache_store = pickle.load(f)

    def _save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self._cache_store, f)

    def _generate_hash(self, video_id: str, query: str) -> str:
        combined_string = f"{video_id}_{query.strip().lower()}"
        return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

    def get_cached_result(self, video_id: str, query: str) -> Optional[Dict[str, Any]]:
        key = self._generate_hash(video_id, query)
        return self._cache_store.get(key)

    def set_cache_result(self, video_id: str, query: str, response_data: Dict[str, Any]) -> None:
        key = self._generate_hash(video_id, query)
        self._cache_store[key] = response_data
        self._save_cache()