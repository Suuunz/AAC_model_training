from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast
from .schema import TrainItem
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..configs.config import DataConfig
    from ..utils.constraints import ChunkOntology

log = get_logger("dataset")

class ChunkDatasetBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        cfg: "DataConfig",
        ontology: "ChunkOntology",
    ):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.onto = ontology
        self._add_special_tokens()

    def _add_special_tokens(self):
        """
        ★ 중요: [SIT], [CTX] 같은 커스텀 특수 토큰을 토크나이저에 추가합니다.
        """
        new_tokens = [self.cfg.situation_token, self.cfg.context_sep_token]
        # 온톨로지의 모든 청크 값도 토큰으로 추가 (선택 사항, 성능 향상 기대)
        # new_tokens.extend(self.onto.all_values) 
        
        # 고유한 새 토큰만 추가
        tokens_to_add = []
        for token in new_tokens:
            if token not in self.tokenizer.vocab:
                tokens_to_add.append(token)
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            log.info(f"Added {len(tokens_to_add)} new special tokens: {tokens_to_add}")
        else:
            log.info("No new special tokens to add.")

    def _encode(self, example: Dict) -> Dict:
        """
        하나의 샘플(dict)을 인코딩합니다.
        입력 포맷: [CLS] [SIT] {상황} [CTX] {점원 발화} [CTX] {사용자 선택 청크} [SEP]
        """
        item = TrainItem(**example)
        
        # V1: " ".join(item.selected_chunks)
        # V2: 청크값 사이에도 [CTX] 토큰을 넣어 경계를 명확히 함
        selected_history = f" {self.cfg.context_sep_token} ".join(item.selected_chunks)
        
        text = (
            f"{self.cfg.situation_token} {item.situation} "
            f"{self.cfg.context_sep_token} {item.staff_utterance} "
            f"{self.cfg.context_sep_token} {selected_history}"
        )
        
        # 토크나이징
        enc = self.tokenizer(
            text,
            max_length=self.cfg.max_length,
            truncation=True,
            padding=False, # DataCollator가 패딩 처리
        )
        
        # 라벨 (청크 값 ID)
        enc["labels"] = self.onto.value2id[item.next_chunk]
        
        # 제약 마스킹에 필요한 원본 청크 리스트 (eval.py용)
        # DataCollator는 이 컬럼을 무시함
        enc["raw_selected_chunks"] = item.selected_chunks 
        return enc

    def build_dataset(self, split: str = "train") -> Dataset:
        """
        JSONL 파일을 로드하고 매핑하여 Hugging Face Dataset 객체를 반환합니다.
        """
        file_path = self.cfg.train_file if split == "train" else self.cfg.valid_file
        
        ds = load_dataset("json", data_files={split: file_path})[split]
        
        # 'raw_selected_chunks'를 제외한 컬럼을 지우도록 설정
        cols_to_remove = [col for col in ds.column_names if col != "raw_selected_chunks"]
        
        ds = ds.map(
            self._encode,
            batched=False,
            remove_columns=cols_to_remove,
        )
        return ds