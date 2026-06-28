import os
import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.config import settings

class LectureSearchResponseSchema(BaseModel):
    answer: str = Field(description="Câu trả lời ngắn gọn bằng tiếng Việt dưới 100 từ.")
    preview: List[str] = Field(description="Mảng chứa 3 đến 5 ý chính tóm tắt nội dung.")
    timestamps: List[float] = Field(description="Mốc thời gian dẫn chứng trực tiếp từ ngữ cảnh.")

class LLMAgent:
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Thiếu GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(settings.LLM_MODEL) 

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_blocks = [f"[{c.get('start_time')} giây]\n{c.get('text')}" for c in contexts]
        context_str = "\n\n".join(context_blocks)

        prompt = f"""
Bạn là chuyên gia phân tích bài giảng video.
QUY TẮC:
- Chỉ sử dụng thông tin trong ngữ cảnh. Không suy diễn.
- Ngắn gọn, không quá 70 từ.

NGỮ CẢNH:
{context_str}

CÂU HỎI:
{query}

Trả về JSON:
{{
  "answer": "...",
  "preview": ["...", "..."],
  "timestamps": [12.5, 55.0]
}}
"""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LectureSearchResponseSchema,
                temperature=0.0,
            )
        )
        return json.loads(response.text)