import logging
from pathlib import Path
import tflux.pipeline.config as config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(logging.DEBUG)  # let handlers filter individually

        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

        # Logs to terminal with config.LOG_LEVEL
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # Logs to file with DEBUG
        if config.LOG_FILE:
            log_path = Path(config.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)    # Keeps all logs
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger