from __future__ import annotations
import json
from typing import Dict, List, Set
from pathlib import Path
import torch

from ..utils.logger import get_logger

log = get_logger("ontology")

class ChunkOntology:
    def __init__(self, path: str | Path):
        log.info(f"Loading ontology from {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        
        self.domain: str = obj["domain"]
        self.slots: Dict[str, List[str]] = obj["slots"]
        self.slot_flow: Dict[str, List[str]] = obj["slot_flow"]
        self.special_values: List[str] = obj.get("special_values", ["<START>", "<END>"])

        # 1. 모든 청크 '값' 리스트 및 매퍼 생성 (이것이 모델의 라벨이 됨)
        all_values_set: Set[str] = set(self.special_values)
        for values in self.slots.values():
            all_values_set.update(values)
        
        self.all_values: List[str] = sorted(list(all_values_set))
        self.value2id: Dict[str, int] = {v: i for i, v in enumerate(self.all_values)}
        self.id2value: Dict[int, str] = {i: v for i, v in enumerate(self.all_values)}

        # 2. '값'이 어느 '슬롯'에 속하는지 역방향 매퍼 생성
        self.value_to_slot: Dict[str, str] = {}
        for slot, values in self.slots.items():
            for value in values:
                if value in self.value_to_slot:
                    log.warning(f"Value '{value}' is in multiple slots. Overwriting.")
                self.value_to_slot[value] = slot
        
        # 특수 값 매핑
        for val in self.special_values:
            if val not in self.value_to_slot:
                self.value_to_slot[val] = val # 예: <START> -> <START>

        log.info(f"Ontology loaded. Domain: {self.domain}, Total Values: {self.num_values()}")

    def num_values(self) -> int:
        return len(self.all_values)

    def allowed_next_values(self, selected_chunks: List[str]) -> List[str]:
        """
        현재까지 선택된 청크 리스트를 기반으로,
        다음에 올 수 있는 '청크 값' 리스트를 반환합니다.
        """
        if not selected_chunks:
            # V1: selected_chunks=["<START>"] 가정
            # V2: 비어있으면 <START>로 간주
            last_chunk = "<START>"
        else:
            last_chunk = selected_chunks[-1]

        # 1. 마지막 청크가 속한 슬롯을 찾음
        last_slot = self.value_to_slot.get(last_chunk)
        if last_slot is None:
            log.warning(f"Last chunk '{last_chunk}' not found in ontology. Defaulting to <START>.")
            last_slot = "<START>"
            
        # 2. 해당 슬롯에서 전이 가능한 *다음 슬롯* 리스트를 찾음
        next_allowed_slots = self.slot_flow.get(last_slot, [])
        
        # 3. 다음 슬롯들에 속한 *모든 값*을 수집
        allowed_values: Set[str] = set()
        for slot in next_allowed_slots:
            allowed_values.update(self.slots.get(slot, []))

        # <END>는 항상 허용 (선택 사항)
        if "<END>" in self.all_values:
             allowed_values.add("<END>")
             
        if not allowed_values:
            # 흐름이 막혔으면 <END>만 허용
            return ["<END>"] if "<END>" in self.all_values else []

        return sorted(list(allowed_values))

    def get_mask(
        self,
        batch_selected_chunks: List[List[str]],
        device: torch.device
    ) -> torch.Tensor:
        """
        배치 단위로 제약 마스크 텐서(-inf, 0)를 생성합니다.
        """
        batch_size = len(batch_selected_chunks)
        mask = torch.full(
            (batch_size, self.num_values()),
            fill_value=float("-inf"),
            device=device
        )

        for i, selected_chunks in enumerate(batch_selected_chunks):
            allowed_values = self.allowed_next_values(selected_chunks)
            if allowed_values:
                allowed_indices = [self.value2id[v] for v in allowed_values if v in self.value2id]
                if allowed_indices:
                    mask[i, allowed_indices] = 0.0
        return mask
        
    def save_mappers(self, save_dir: Path):
        """추론 시 id<->value 매핑을 위해 파일 저장"""
        with open(save_dir / "value2id.json", "w", encoding="utf-8") as f:
            json.dump(self.value2id, f, ensure_ascii=False, indent=2)
        with open(save_dir / "id2value.json", "w", encoding="utf-8") as f:
            json.dump(self.id2value, f, ensure_ascii=False, indent=2)