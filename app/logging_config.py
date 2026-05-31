import logging
import sys

def configure_logging(level: int = logging.INFO) -> None:
    """Configures systematic logging for the backend application."""
    # Standard format: [Timestamp] [Level] [LoggerName] Message
    log_format = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Check if colorama is available or support standard ANSI color coding
    class ColoredFormatter(logging.Formatter):
        GREY = "\x1b[38;20m"
        YELLOW = "\x1b[33;20m"
        GREEN = "\x1b[32;20m"
        RED = "\x1b[31;20m"
        BOLD_RED = "\x1b[31;1m"
        CYAN = "\x1b[36;20m"
        RESET = "\x1b[0m"

        LEVEL_COLORS = {
            logging.DEBUG: GREY,
            logging.INFO: CYAN,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: BOLD_RED
        }

        def format(self, record):
            log_fmt = self.LEVEL_COLORS.get(record.levelno, self.RESET) + log_format + self.RESET
            formatter = logging.Formatter(log_fmt, datefmt=date_format)
            return formatter.format(record)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    
    # Quiet noisy external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logging.getLogger("app").info("Application logging successfully configured.")
