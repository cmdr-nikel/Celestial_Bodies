import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')


def setup_logger(script_name: str) -> logging.Logger:
    """
    Creates a logger for the given script name.
    Writes to both console and a timestamped log file in backend/logs/.
    Keeps a maximum of 10 log files per script (oldest are deleted automatically).

    Usage:
        from utils.logger import setup_logger
        logger = setup_logger("01_eda")
        logger.info("Starting EDA...")
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp}_{script_name}.log"
    log_filepath = os.path.join(LOGS_DIR, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if logger is re-initialised in a notebook
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- File handler (rotating: max 10 files per script) ---
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Clean up old logs for this script (keep only 10 most recent)
    _cleanup_old_logs(script_name, keep=10)

    logger.info(f"Logger initialised | script={script_name} | log={log_filename}")
    return logger


def _cleanup_old_logs(script_name: str, keep: int = 10):
    """Deletes oldest log files for this script, keeping only `keep` most recent."""
    all_logs = sorted([
        f for f in os.listdir(LOGS_DIR)
        if f.endswith(f"_{script_name}.log")
    ])
    if len(all_logs) > keep:
        for old_file in all_logs[:len(all_logs) - keep]:
            try:
                os.remove(os.path.join(LOGS_DIR, old_file))
            except OSError:
                pass