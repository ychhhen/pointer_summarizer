import logging
import os
import sys
import time

from pathlib2 import Path

from data_util.config import log_file, log_level

LOGGER = None


def get_logging_level():
    log_levels = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARN': logging.WARNING
    }
    return log_levels.get(log_level, logging.INFO)


def configure_logger(name, run_type='training'):
    # Create log directory
    log_dir = Path(os.path.dirname(log_file))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configure file handler
    file_handler = logging.FileHandler(
        log_file.format(run_type, int(time.time())))
    file_handler.setLevel(get_logging_level())
    file_handler.setFormatter(formatter)

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(get_logging_level())
    console_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(get_logging_level())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name, run_type='training'):
    global LOGGER
    if not LOGGER:
        LOGGER = configure_logger(name, run_type)
    return LOGGER
