from rich.console import Console
from rich.logging import RichHandler
import logging
import os

_console = Console(color_system="auto")

def get_logger(name: str = "aac") -> logging.Logger:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=_console, 
            markup=True, 
            rich_tracebacks=True,
            show_path=False # 경로 간소화
        )]
    )
    logger = logging.getLogger(name)
    logger.propagate = False # 중복 로깅 방지
    return logger