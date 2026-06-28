import os
from typing import List, Dict, Any
from ..vector_store.faiss_store import FAISSStore 

class DenseStore:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.vector_store = FAISSStore(embedding_dim=1024)
        self.encoder = self.vector_store.model

    def build_index(self, chunks: List[Dict[str, Any]], video_id: str = "default_video") -> None:
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.encoder.encode(
            texts, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        self.vector_store.add_chunks(chunks, embeddings, video_id=video_id)

    def save_index(self, directory: str, index_name: str = "faiss_index", video_id: str = None) -> None:
        actual_video_id = video_id if video_id else index_name
        self.vector_store.save_index(directory, index_name, video_id=actual_video_id)

    def load_index(self, directory: str, index_name: str = "faiss_index") -> None:
        self.vector_store.load_index(directory, index_name)

    def search(self, query: str, top_k: int = 5, video_id: str = None) -> List[Dict[str, Any]]:
        if not video_id:
            raise ValueError("Bắt buộc phải cung cấp 'video_id' để tìm kiếm.")

        query_vector = self.encoder.encode(
            query, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        return self.vector_store.search(query_vector, top_k=top_k, video_id=video_id)