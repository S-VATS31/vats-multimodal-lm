import logging
import os

# Set up directory for log files to go
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def setup_logger(name: str, log_file: str, level: int) -> logging.Logger:
    """Set up a logger with specified name, file, and level.
    
    Args:
        name (str): Logger name.
        log_file (str): File path for log to be directed to.
        level (int): Level of logger messages.

    Returns:
        logger.Logger: Logger item to be used in other modules.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is reused
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, log_file))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
    