import logging
import logging.config
import os
import json
import sys


def setup_logging(config_path="cfg/config.json", default_level=logging.INFO) -> None:
    """Initialize logging system for the entire application.
    
    Reads logging configuration from config.json, creates necessary log directories,
    and configures both file and console handlers with appropriate formatters.
    
    Args:
        config_path: Path to configuration file (default: cfg/config.json).
        default_level: Default logging level if not specified in config.
    """
    # Load configuration from file
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            log_cfg = config.get("logging", {})
    else:
        log_cfg = {}

    # Set log directory, file, and levels from config
    log_dir = log_cfg.get("log_dir", "logs")
    log_file = log_cfg.get("log_file", "app.log")
    file_level = getattr(logging, log_cfg.get("file_level", "DEBUG"))
    console_level = getattr(logging, log_cfg.get("console_level", "INFO"))
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Clear previous handlers to avoid duplicate logs if called multiple times
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    # Configure formatters
    # File formatter: detailed with timestamp, logger name, level, and message
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console formatter: minimal output (message only)
    console_formatter = logging.Formatter("%(message)s")

    # Configure handlers
    
    # File handler: logs all messages with detailed formatting
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler: logs important messages with minimal formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initialization confirmation
    logging.getLogger(__name__).debug(f"Logging initialized. File: {log_path}")