
import os
import shutil
from pathlib import Path
from functools import partial
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.utils import logging as hf_logging

from .configs.config import load_config
from .utils.logger import get_logger
from .utils.constraints import ChunkOntology
from .data.dataset import ChunkDatasetBuilder
from .models.predictor import ChunkPredictor

# Hugging Face 로거 레벨 설정
hf_logging.set_verbosity_info()
log = get_logger("train")

def compute_metrics(eval_pred, ontology: ChunkOntology, top_k: int = 5):
    """
    평가 메트릭을 계산합니다.
    여기서는 *제약 없는* Top-k 정확도를 계산합니다.
    (제약이 적용된 정확도는 eval.py 참조)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Top-1 정확도
    top1_acc = np.mean(preds == labels)
    
    # Top-k 정확도 (k=5)
    topk_preds = np.argsort(logits, axis=-1)[:, -top_k:]
    topk_acc = np.mean([labels[i] in topk_preds[i] for i in range(len(labels))])
    
    return {
        f"top1_accuracy_unconstrained": top1_acc,
        f"top{top_k}_accuracy_unconstrained": topk_acc,
    }

def train_pipeline(config_path: str):
    log.info(f"Loading config from: [yellow]{config_path}[/yellow]")
    cfg = load_config(config_path)

    # 출력 디렉터리 설정 (기존 폴더 덮어쓰기)
    output_dir = Path(cfg.model.output_dir)
    if output_dir.exists() and cfg.train.overwrite_output_dir:
        log.warning(f"Overwriting output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 온톨로지 로드
    log.info(f"Loading ontology from: {cfg.data.ontology_file}")
    onto = ChunkOntology(cfg.data.ontology_file)
    onto.save_mappers(output_dir) # value2id, id2value 저장

    # 2. 토크나이저 로드
    log.info(f"Loading tokenizer: {cfg.model.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    
    # 3. 데이터셋 빌더 및 특수 토큰 추가
    # ★ 중요: 특수 토큰을 추가하고 모델 임베딩을 리사이징합니다.
    log.info("Initializing DatasetBuilder and adding special tokens...")
    ds_builder = ChunkDatasetBuilder(
        tokenizer=tokenizer,
        cfg=cfg.data,
        ontology=onto
    )
    
    # 모델 로드 (임베딩 리사이징 포함)
    log.info("Loading model and resizing token embeddings...")
    model = ChunkPredictor.load(
        model_name=cfg.model.base_model_name,
        num_labels=onto.num_values(),
    )
    model.resize_token_embeddings(len(ds_builder.tokenizer))
    log.info(f"Tokenizer size: {len(ds_builder.tokenizer)}, Model output labels: {onto.num_values()}")

    # 4. 데이터셋 빌드 (학습/검증)
    log.info("Building datasets...")
    train_ds = ds_builder.build_dataset("train")
    valid_ds = ds_builder.build_dataset("validation")
    log.info(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        report_to=cfg.train.report_to,
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        fp16=cfg.train.fp16,
        logging_steps=cfg.train.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        seed=cfg.project.seed,
    )

    # 6. 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=partial(compute_metrics, ontology=onto, top_k=cfg.inference.top_k),
    )

    # 7. 학습 시작
    log.info("[bold green]Starting training...[/bold green]")
    trainer.train()

    # 8. 모델 및 토크나이저 저장
    log.info("Training finished. Saving best model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # 설정 파일 복사
    shutil.copy(config_path, output_dir / "config.yaml")

    log.info(f"[bold green]Training complete. Model saved to {output_dir}[/bold green]")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="aac_chunks/configs/config.yaml")
    args = parser.parse_args()
    
    train_pipeline(args.config)