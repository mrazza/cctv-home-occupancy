import logging
import os
import sys
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = "logs/cctv.log"):
    """Configures standard logging to both stdout and optionally a file."""
    # Convert string log level to numeric
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Create formatters
    log_format = "[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(name)s:%(lineno)d]: %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers to prevent duplicate logging
    root_logger.handlers = []

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Silencing noise from standard third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    
    # Silence httpx logs unless level is explicitly set to DEBUG
    if numeric_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
