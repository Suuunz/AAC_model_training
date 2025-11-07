from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn

from ..configs.config import load_config, AppConfig
from ..infer import ChunkInfer
from ..data.schema import InferRequest, InferResponse
from ..utils.logger import get_logger

# --- 전역 변수 ---
# 실제 프로덕션에서는 DI(Dependency Injection) 프레임워크 사용 권장
cfg: AppConfig | None = None
infer_engine: ChunkInfer | None = None
log = get_logger("api")

# --- FastAPI 라이프사이클 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 로드
    global cfg, infer_engine
    try:
        config_path = Path("aac_chunks/configs/config.yaml")
        log.info(f"Loading config from {config_path}")
        cfg = load_config(config_path)
        
        model_dir = Path(cfg.model.output_dir)
        log.info(f"Loading inference engine from {model_dir}...")
        infer_engine = ChunkInfer(model_dir=model_dir, config=cfg)
        log.info("[bold green]Inference engine loaded successfully.[/bold green]")
    except Exception as e:
        log.error(f"Failed to load model on startup: {e}", exc_info=True)
        infer_engine = None # 로드 실패 명시
    
    yield
    
    # 종료 시
    log.info("Shutting down API.")

# --- 의존성 주입 ---
def get_engine() -> ChunkInfer:
    if infer_engine is None:
        log.error("Inference engine is not initialized.")
        raise HTTPException(status_code=503, detail="Server is not ready, model not loaded.")
    return infer_engine

def get_config() -> AppConfig:
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration is not loaded.")
    return cfg

# --- FastAPI 앱 ---
app = FastAPI(
    title="AAC ChunkNext API (V2 Refactored)",
    description="상황, 점원 발화, 선택 이력을 기반으로 다음 청크 값을 추천합니다.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    if infer_engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/recommend", response_model=InferResponse)
def recommend(
    req: InferRequest,
    engine: ChunkInfer = Depends(get_engine),
    config: AppConfig = Depends(get_config)
):
    """
    다음 청크(chunk) 후보를 Top-k로 추천합니다.
    """
    try:
        top_k = req.top_k or config.inference.top_k
        
        labels, scores = engine.recommend(
            situation=req.situation,
            staff_utterance=req.staff_utterance,
            selected_chunks=req.selected_chunks,
            top_k=top_k
        )
        
        # (디버깅/UI용) 현재 허용된 모든 값 목록
        allowed_values = engine.get_allowed_values(req.selected_chunks)
        
        return InferResponse(
            candidates=labels,
            scores=scores,
            allowed_values=allowed_values
        )
    except Exception as e:
        log.error(f"Error during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    # 실행: uvicorn aac_chunks.api.server:app --reload --port 8000
    uvicorn.run(
        "aac_chunks.api.server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )