import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class SemanticChunker:
    def __init__(self, model_name: str = "BAAI/bge-m3", percentile_threshold: float = 90.0, max_words: int = 500):
        self.encoder = SentenceTransformer(model_name)
        self.percentile_threshold = percentile_threshold
        self.max_words = max_words

    def _get_word_count(self, text_list: List[str]) -> int:
        return len(" ".join(text_list).split())

    def chunk_sentences(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [{
                "text": sentences[0]["text"],
                "start_time": sentences[0].get("start_time"),
                "end_time": sentences[0].get("end_time"),
                "video_id": sentences[0].get("video_id", "unknown")
            }]

        texts = [s["text"] for s in sentences]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        
        emb_current = embeddings[:-1]
        emb_next = embeddings[1:]
        
        similarities = np.sum(emb_current * emb_next, axis=1)
        distances = 1.0 - similarities

        threshold = np.percentile(distances, self.percentile_threshold)
        
        if threshold <= 0.0:
            threshold = 0.2 
        chunks = []
        current_chunk_texts = [sentences[0]["text"]]
        current_start_time = sentences[0].get("start_time")

        for i in range(len(distances)):
            current_word_count = self._get_word_count(current_chunk_texts + [sentences[i+1]["text"]])
            
            if distances[i] <= threshold and current_word_count <= self.max_words:
                current_chunk_texts.append(sentences[i+1]["text"])
            else:
                chunks.append({
                    "text": " ".join(current_chunk_texts),
                    "start_time": current_start_time,
                    "end_time": sentences[i].get("end_time"),
                    "video_id": sentences[i].get("video_id", "unknown")
                })
                current_start_time = sentences[i+1].get("start_time")
                current_chunk_texts = [sentences[i+1]["text"]]

        if current_chunk_texts:
            chunks.append({
                "text": " ".join(current_chunk_texts),
                "start_time": current_start_time,
                "end_time": sentences[-1].get("end_time"),
                "video_id": sentences[-1].get("video_id", "unknown")
            })

        return chunks