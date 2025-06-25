"""
logger for the project
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    
    formatter = logging.Formatter(format_string)

    #create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #prevent duplicate handlers
    if logger.handlers:
        return logger

    #create console handler and set level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
    
    
