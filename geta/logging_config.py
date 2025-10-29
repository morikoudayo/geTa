"""
Logging configuration with uvicorn-compatible colored output
"""

import logging


class ColoredFormatter(logging.Formatter):
    """Uvicorn-style colored log formatter"""

    COLORS = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> logging.Logger:
    """Setup logging with colored formatter matching uvicorn style"""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(levelname)s:     %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return logging.getLogger(__name__)
