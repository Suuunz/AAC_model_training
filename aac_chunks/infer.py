from __future__ import annotations
import json
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizerFast
from pathlib import Path
from .utils.constraints import ChunkOntology
from .configs.config import load_config, AppConfig, DataConfig

class ChunkInfer:
    def __init__(
        self,
        model_dir: str | Path,
        config: AppConfig,
        device: str | None = None
    ):
        self.model_dir = Path(model_dir)
        self.cfg = config
        self.data_cfg: DataConfig = config.data # 타입 명시
        
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 1. 온톨로지 로드
        self.onto = ChunkOntology(self.data_cfg.ontology_file)
        
        # 2. 토크나이저 및 모델 로드
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(self.device)

    def _encode_input(
        self, 
        situation: str, 
        staff_utterance: str, 
        selected_chunks: List[str]
    ) -> dict:
        """추론용 입력을 인코딩합니다 (배치 크기 1)"""
        if not selected_chunks:
            selected_chunks = ["<START>"] # V2 로직: 비어있으면 <START>
            
        selected_history = f" {self.data_cfg.context_sep_token} ".join(selected_chunks)
        
        text = (
            f"{self.data_cfg.situation_token} {situation} "
            f"{self.data_cfg.context_sep_token} {staff_utterance} "
            f"{self.data_cfg.context_sep_token} {selected_history}"
        )
        
        enc = self.tok(
            text,
            max_length=self.data_cfg.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def recommend(
        self,
        situation: str,
        staff_utterance: str,
        selected_chunks: List[str],
        top_k: int
    ) -> Tuple[List[str], List[float]]:
        
        # 1. 입력 인코딩
        batch = self._encode_input(situation, staff_utterance, selected_chunks)
        
        # 2. 모델 추론
        logits = self.model(**batch).logits
        
        # 3. 제약 마스킹 적용 (핵심)
        # (배치 크기 1)
        if not selected_chunks:
            selected_chunks = ["<START>"]
            
        constraint_mask = self.onto.get_mask(
            batch_selected_chunks=[selected_chunks],
            device=self.device
        )
        masked_logits = logits + constraint_mask
        
        # 4. Top-k 추출
        probs = torch.softmax(masked_logits, dim=-1)[0]
        topk_result = torch.topk(probs, k=top_k)
        
        scores = [float(s) for s in topk_result.values]
        indices = [int(i) for i in topk_result.indices]
        labels = [self.onto.id2value[i] for i in indices]
        
        # 점수가 0인 (즉, -inf로 마스킹된) 후보는 제거
        valid_labels = []
        valid_scores = []
        for label, score in zip(labels, scores):
            if score > 1e-9: # 0에 가까운 값 제외
                valid_labels.append(label)
                valid_scores.append(score)
                
        return valid_labels, valid_scores

    def get_allowed_values(self, selected_chunks: List[str]) -> List[str]:
        """(디버깅/UI용) 현재 상태에서 허용된 모든 값 반환"""
        if not selected_chunks:
            selected_chunks = ["<START>"]
        return self.onto.allowed_next_values(selected_chunks)