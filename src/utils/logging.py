
import logging
from loguru import logger

"""
# Example of how to use loguru with logging
import sys
import loguru

logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time:%H:%M:%S.%f}<level>[{level}]: {message}</level>", colorize=True)

logger.add("logs/transcription_server.log", level="DEBUG", format="{time:%H:%M:%S.%f}[{level}]: {message}")

"""

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def add_other_loggers(other_modules: list, level="DEBUG"):

    for module in other_modules:
        logging.getLogger(module).handlers = [InterceptHandler()]
        logging.getLogger(module).setLevel(getattr(logging, level))
    
