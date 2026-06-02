import os
import logging
from src.logger import setup_logging

def test_setup_logging_console_and_file(temp_dir):
    log_file = os.path.join(temp_dir, "test_cctv_log.log")
    setup_logging(log_level="DEBUG", log_file=log_file)
    
    logger = logging.getLogger("test_logger")
    logger.debug("Testing debug message")
    logger.info("Testing info message")
    
    # Assert log file was created and contains the logs
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
    assert "Testing debug message" in content
    assert "Testing info message" in content

def test_setup_logging_no_file():
    setup_logging(log_level="WARNING", log_file=None)
    root_logger = logging.getLogger()
    
    # Root logger should have a Console handler (StreamHandler) but no FileHandler
    has_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            has_stream_handler = True
        assert not isinstance(handler, logging.FileHandler)
    assert has_stream_handler

def test_setup_logging_invalid_level():
    # Should fall back to INFO
    setup_logging(log_level="INVALID_LEVEL", log_file=None)
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
