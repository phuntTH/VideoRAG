import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
import pickle
import logging
import threading
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.services.vector_store.base import BaseVectorStore
from app.config import settings # <--- Import settings

logger = logging.getLogger("video_rag_backend")

class FAISSStore(BaseVectorStore):
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._active_indices = {}
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, embedding_dim: int = 1024):
        if self._initialized:
            return
            
        self.embedding_dim = embedding_dim
        self.default_base_dir = str(settings.FAISS_INDICES_DIR) 
        self._initialized = True
        logger.info(f"💾 FAISSStore Engine initialized (Dimension: {self.embedding_dim})")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🚀 [Lazy Loading] Initializing global BGE-M3 on [{device.upper()}]...")
            try:
                self._model = SentenceTransformer("BAAI/bge-m3", device=device)
            except Exception as e:
                logger.error(f"Fatal error loading embedding layers: {e}")
                raise e
        return self._model

    def _ensure_video_index_loaded(self, video_id: str, custom_dir: str = None) -> None:
        if video_id in self._active_indices:
            return

        with self._lock:
            if video_id in self._active_indices:
                return

            base_path = custom_dir if custom_dir else self.default_base_dir
            directory = os.path.join(base_path, video_id)
            index_path = os.path.join(directory, "dense_index.faiss")
            meta_path = os.path.join(directory, "metadata.pkl")

            if os.path.exists(index_path) and os.path.exists(meta_path):
                logger.info(f"📁 Index found on disk for video [{video_id}]. Loading to RAM...")
                index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                
                self._active_indices[video_id] = {"index": index, "metadata": metadata}
            else:
                logger.info(f"✨ Creating fresh vector index context for video [{video_id}].")
                self._active_indices[video_id] = {
                    "index": faiss.IndexFlatIP(self.embedding_dim),
                    "metadata": []
                }

    def build_index(self, chunks: List[Dict[str, Any]], video_id: str = "default_video") -> None:
        logger.info(f"🧱 Vectorizing {len(chunks)} text chunks for video: {video_id}")
        self._ensure_video_index_loaded(video_id)
        
        texts = [chunk.get("text", "") for chunk in chunks]
        if not texts:
            return

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        with self._lock:
            target = self._active_indices[video_id]
            target["index"].add(embeddings)
            target["metadata"].extend(chunks)

        directory = os.path.join(self.default_base_dir, video_id)
        self.save_index(directory=directory, video_id=video_id)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, video_id: str = "default_video") -> None:
        self._ensure_video_index_loaded(video_id)
        with self._lock:
            target = self._active_indices[video_id]
            target["index"].add(embeddings.astype('float32'))
            target["metadata"].extend(chunks)
            
        directory = os.path.join(self.default_base_dir, video_id)
        self.save_index(directory=directory, video_id=video_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 5, video_id: str = None) -> List[Dict[str, Any]]:
        if not video_id:
            raise ValueError("Bắt buộc phải cung cấp 'video_id'.")
        self._ensure_video_index_loaded(video_id)
        
        target = self._active_indices[video_id]
        index = target["index"]
        metadata = target["metadata"]

        if index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(metadata):
                result_item = metadata[idx].copy()
                result_item["score"] = float(dist)
                results.append(result_item)
        return results

    def save_index(self, directory: str, index_name: str = "dense_index", video_id: str = None) -> None:
        if not video_id or video_id not in self._active_indices:
            return

        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, f"{index_name}.faiss")
        meta_path = os.path.join(directory, f"{index_name.replace('dense_index', 'metadata')}.pkl")

        with self._lock:
            target = self._active_indices[video_id]
            faiss.write_index(target["index"], index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(target["metadata"], f)

    def load_index(self, directory: str, index_name: str = "dense_index") -> None:
        video_id = index_name if index_name and index_name != "dense_index" else os.path.basename(os.path.normpath(directory))
        self._ensure_video_index_loaded(video_id, custom_dir=directory)