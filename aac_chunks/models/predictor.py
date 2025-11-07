from __future__ import annotations
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

class ChunkPredictor:
    """
    klue/roberta-base 기반의 Sequence Classification 모델 래퍼
    """
    @staticmethod
    def load(model_name: str, num_labels: int) -> PreTrainedModel:
        """
        지정된 num_labels로 모델을 로드합니다.
        """
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            # 로컬 가중치가 아닌 Hugging Face Hub의 사전학습 가중치 사용 보장
            ignore_mismatched_sizes=True 
        )
        return model