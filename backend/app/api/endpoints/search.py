from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import time

from ...config import settings
from ...services.retrieval.dense_search import DenseStore
from ...services.retrieval.sparse_search import BM25Store
from ...services.retrieval.rrf_fusion import reciprocal_rank_fusion
from ...services.generation.llm_agent import LLMAgent
from ...core.cache import PersistentSearchCache

router = APIRouter()
query_cache = PersistentSearchCache()

llm_agent_engine = LLMAgent()

class VideoIndexRegistry:
    def __init__(self):
        self._dense_indices = {}
        self._sparse_indices = {}

    def get_indices(self, video_id: str):
        if video_id not in self._dense_indices:
            print(f"Loading {video_id} indices into RAM for the first time...")
            dense = DenseStore()
            dense.load_index(settings.FAISS_INDICES_DIR, index_name=video_id)
            
            sparse = BM25Store()
            sparse.load_index(settings.BM25_INDICES_DIR, index_name=video_id)
            
            self._dense_indices[video_id] = dense
            self._sparse_indices[video_id] = sparse
            
        return self._dense_indices[video_id], self._sparse_indices[video_id]

index_registry = VideoIndexRegistry()

class SearchQueryRequest(BaseModel):
    video_id: str
    query: str

@router.post("")
async def execute_hybrid_search(request: SearchQueryRequest):
    start_time = time.time()
    
    cached_data = query_cache.get_cached_result(request.video_id, request.query)
    if cached_data:
        return cached_data

    try:
        dense_store, sparse_store = index_registry.get_indices(request.video_id)

        # 3. Dense Search
        query_vector = dense_store.encoder.encode(request.query, normalize_embeddings=True, convert_to_numpy=True)
        dense_res = dense_store.vector_store.search(query_vector, top_k=20, video_id=request.video_id)

        # 4. Sparse Search (Đã sửa lỗi gán trường dữ liệu 'score')
        tokenized_query = sparse_store._tokenize(request.query)
        bm25_scores = sparse_store.bm25.get_scores(tokenized_query)
        top_sparse_indices = bm25_scores.argsort()[::-1][:20]
        
        sparse_res = []
        for idx in top_sparse_indices:
            if bm25_scores[idx] > 0:
                chunk_data = sparse_store.metadata[idx].copy()
                chunk_data["score"] = float(bm25_scores[idx]) # <--- Khắc phục thiếu score đồng bộ hệ thống
                sparse_res.append(chunk_data)

        # 5. RRF Fusion
        fused_chunks = reciprocal_rank_fusion(dense_res, sparse_res, top_n=20)

        # 6. Bypass Rerank - Lấy trực tiếp Top 5 từ kết quả thuật toán RRF cực kỳ tối ưu
        final_context_chunks = fused_chunks[:5]

        print("\n" + "=" * 80)
        print("FINAL CONTEXT SENT TO LLM (WITHOUT RERANK ENGINE)")
        print("=" * 80)
        for i, chunk in enumerate(final_context_chunks):
            print(f"Chunk {i+1} | Start: {chunk.get('start_time')}s | Score: {chunk.get('score', 0.0)}")

        # 7. LLM Generation
        llm_response = llm_agent_engine.generate_response(request.query, final_context_chunks)
        
        if "timestamps" in llm_response and llm_response["timestamps"]:
            llm_response["timestamps"] = sorted(list(set(llm_response["timestamps"])))
        
        combined_payload = {
            "answer": llm_response.get("answer"),
            "preview": [c.get("text", "")[:150] + "..." for c in final_context_chunks],
            "timestamps": [c.get("start_time", 0.0) for c in final_context_chunks],
            "rerank_scores": [c.get("score", 0.0) for c in final_context_chunks],
            "contexts": final_context_chunks 
        }
        
        # 8. Save to cache
        query_cache.set_cache_result(request.video_id, request.query, combined_payload)
        
        print(f"⏱️ Done in: {time.time() - start_time:.4f}s")
        return combined_payload

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")