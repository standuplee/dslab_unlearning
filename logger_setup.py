import logging
import logging.config
import os
import wandb

def setup_logger(log_file):
    """로거 세팅 (콘솔: DEBUG 이상, 파일: INFO 이상)"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {  # 콘솔: DEBUG 이상 출력
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "DEBUG"
            },
            "file": {  # 파일: INFO 이상 기록
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": log_file,
                "level": "INFO",
                "encoding": "utf-8"
            }
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "DEBUG"
        }
    }

    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)

def init_wandb(args):
    """wandb 초기화"""
    wandb.init(
        project="cure4rec-instancewise",
        config=vars(args)
    )
    logging.getLogger(__name__).info("wandb initialized successfully")