import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger("video_rag_backend")

class BGEReranker:
    _instance = None  

    def __new__(cls, *args, **kwargs):
        """
        Áp dụng cấu trúc Singleton Pattern. 
        Đảm bảo chỉ có duy nhất một đối tượng điều phối Reranker được khởi tạo trong RAM.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None  
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        if self._initialized:
            return 
            
        self.model_name = model_name
        self._initialized = True
        logger.info(f"💾 BGEReranker meta-object proxy initialized using model framework: {self.model_name} (Weights deferred)")

    @property
    def model(self) -> CrossEncoder:
        """
        Cơ chế Lazy Loading lõi. 
        Mô hình CrossEncoder chỉ thực sự được nạp lên RAM khi thuộc tính này được truy cập lần đầu.
        """
        if self._model is None:
            logger.info(f"🚀 [Lazy Loading] Triggered! Loading CrossEncoder model weights for {self.model_name} into memory...")
            try:
                self._model = CrossEncoder(self.model_name)
                logger.info("✅ CrossEncoder model weights successfully bound to active memory workspace.")
            except Exception as e:
                logger.error(f"Fatal crash allocating system resources to load cross-encoder model partitions: {e}")
                raise e
        return self._model

    def predict(self, pairs: list) -> list:
        """
        Thực hiện tính toán điểm số Relevance Score giữa câu hỏi và các đoạn văn bản context.
        """
        if not pairs:
            return []
        return self.model.predict(pairs)