from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# Pydantic 모델 정의
class ProjectConfig(BaseModel):
    seed: int = 42

class DataConfig(BaseModel):
    train_file: str
    valid_file: str
    ontology_file: str
    max_length: int = 192
    situation_token: str = "[SIT]"
    context_sep_token: str = "[CTX]"

class ModelConfig(BaseModel):
    base_model_name: str = "klue/roberta-base"
    output_dir: str = "outputs/aac-roberta-v1"

class TrainConfig(BaseModel):
    overwrite_output_dir: bool = True
    report_to: str = "none"
    num_train_epochs: int = 5
    learning_rate: float = 5.0e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_top1_accuracy_unconstrained"
    greater_is_better: bool = True
    lr_scheduler_type: str = "linear"

class InferConfig(BaseModel):
    top_k: int = 5

# 메인 설정 클래스
class AppConfig(BaseSettings):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    inference: InferConfig = Field(default_factory=InferConfig)

    class Config:
        # YAML 파일에서 로드하기 위한 커스텀 로더
        @classmethod
        def yaml_config_settings_source(cls, settings: BaseSettings) -> dict:
            config_path = getattr(settings.model_config, 'config_path', 'config.yaml')
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

# 설정 로드 함수
def load_config(config_path: str | Path) -> AppConfig:
    return AppConfig(
        _settings_config=SettingsConfigDict(
            config_path=str(config_path),
            # .env 파일 로드 (선택 사항)
            # env_file='.env', 
            # env_nested_delimiter='__'
        )
    )

if __name__ == "__main__":
    # 테스트용
    cfg = load_config(Path(__file__).parent / "config.yaml")
    print(cfg.model_dump_json(indent=2))