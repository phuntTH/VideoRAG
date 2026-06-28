import pickle
import os
import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.metadata: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
            
        self.metadata = chunks
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self, directory: str, index_name: str = "bm25_index") -> None:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{index_name}.pkl")
        
        with open(file_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25, 
                "metadata": self.metadata
            }, f)

    def load_index(self, directory: str, index_name: str = "bm25_index") -> None:
        file_path = os.path.join(directory, f"{index_name}.pkl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 index not found at {file_path}")
            
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.metadata = data["metadata"]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("BM25 index has not been loaded or built.")
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in scored_indices:
            chunk_data = self.metadata[idx].copy()
            chunk_data["score"] = float(score)
            results.append(chunk_data)
            
        return results