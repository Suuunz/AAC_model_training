import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from tqdm.auto import tqdm
import numpy as np

from .configs.config import load_config
from .utils.logger import get_logger
from .utils.constraints import ChunkOntology
from .data.dataset import ChunkDatasetBuilder

log = get_logger("eval")

def evaluate_pipeline(config_path: str, model_dir: str | None = None):
    cfg = load_config(config_path)
    
    # 학습된 모델 경로
    model_path = Path(model_dir or cfg.model.output_dir)
    if not model_path.exists():
        log.error(f"Model directory not found: {model_path}")
        return

    log.info(f"Evaluating model from: [yellow]{model_path}[/yellow]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 온톨로지, 토크나이저, 모델 로드
    onto = ChunkOntology(cfg.data.ontology_file)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # 2. 검증 데이터셋 로드
    ds_builder = ChunkDatasetBuilder(tokenizer=tok, cfg=cfg.data, ontology=onto)
    valid_ds: Dataset = ds_builder.build_dataset("validation")
    
    # 3. 데이터로더 생성 (raw_selected_chunks를 위한 커스텀 collate)
    def collate_fn(features):
        # DataCollator가 처리할 부분
        batch = tok.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            return_tensors="pt"
        )
        # 라벨
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        # 마스킹에 필요한 원본 청크
        batch["raw_selected_chunks"] = [f["raw_selected_chunks"] for f in features]
        return batch

    dataloader = DataLoader(
        valid_ds,
        batch_size=cfg.train.per_device_eval_batch_size,
        collate_fn=collate_fn
    )

    all_labels = []
    all_preds_constrained = []
    all_topk_constrained_hits = []
    top_k = cfg.inference.top_k

    log.info(f"Running evaluation on {len(valid_ds)} samples (Batch size: {cfg.train.per_device_eval_batch_size})...")

    # 4. 평가 루프
    for batch in tqdm(dataloader, desc="Evaluating"):
        labels = batch.pop("labels").numpy()
        raw_chunks = batch.pop("raw_selected_chunks")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(**batch).logits
            
            # 제약 마스크 적용
            constraint_mask = onto.get_mask(raw_chunks, device)
            masked_logits = logits + constraint_mask
            
            # Constrained Top-1
            preds_constrained = torch.argmax(masked_logits, dim=-1).cpu().numpy()
            
            # Constrained Top-k
            topk_indices = torch.topk(masked_logits, k=top_k, dim=-1).indices.cpu().numpy()
            
            topk_hits = [labels[i] in topk_indices[i] for i in range(len(labels))]

            all_labels.extend(labels)
            all_preds_constrained.extend(preds_constrained)
            all_topk_constrained_hits.extend(topk_hits)

    # 5. 결과 집계
    all_labels = np.array(all_labels)
    all_preds_constrained = np.array(all_preds_constrained)
    
    top1_acc = np.mean(all_labels == all_preds_constrained)
    topk_acc = np.mean(all_topk_constrained_hits)

    log.info("[bold green]Evaluation Complete[/bold green]")
    log.info(f"Total Samples: {len(all_labels)}")
    log.info(f"Constrained Top-1 Accuracy: [bold]{top1_acc:.4f}[/bold]")
    log.info(f"Constrained Top-{top_k} Accuracy: [bold]{topk_acc:.4f}[/bold]")

    return {"top1_constrained": top1_acc, f"top{top_k}_constrained": topk_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="aac_chunks/configs/config.yaml")
    parser.add_argument("--model_dir", type=str, default=None, help="Optional: Path to saved model. Defaults to config output_dir.")
    args = parser.parse_args()
    
    evaluate_pipeline(args.config, args.model_dir)