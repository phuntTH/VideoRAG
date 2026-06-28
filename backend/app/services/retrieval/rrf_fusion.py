import hashlib
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60, 
    top_n: int = 20
) -> List[Dict[str, Any]]:
    rrf_scores = {}
    chunk_mapping = {}
    
    def calculate_rrf(results: List[Dict[str, Any]]):
        for rank, chunk in enumerate(results):
            text_bytes = chunk.get("text", "").encode("utf-8")
            chunk_hash = hashlib.md5(text_bytes).hexdigest()
            chunk_id = f"{chunk.get('video_id')}_{chunk.get('start_time')}_{chunk_hash}"
            
            if chunk_id not in chunk_mapping:
                chunk_mapping[chunk_id] = chunk
            
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + (rank + 1)))

    calculate_rrf(dense_results)
    calculate_rrf(sparse_results)
    
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_mapping[chunk_id] for chunk_id, _ in sorted_chunks[:top_n]]