from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class TrainItem(BaseModel):
    """학습 데이터(JSONL)의 한 줄을 나타내는 스키마"""
    dialog_id: str
    situation: str
    staff_utterance: str
    selected_chunks: List[str] = Field(default_factory=list)
    next_chunk: str # 정답 청크 (예: "아이스", "아메리카노")

class InferRequest(BaseModel):
    """추론 API 요청 스키마"""
    situation: str
    staff_utterance: str
    selected_chunks: List[str] = Field(default_factory=list)
    top_k: Optional[int] = None

class InferResponse(BaseModel):
    """추론 API 응답 스키마"""
    candidates: List[str]
    scores: List[float]
    allowed_values: List[str]