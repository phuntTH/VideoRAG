from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def save_index(self, path: str) -> None:
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        pass