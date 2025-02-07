
import logging
from loguru import logger

import time

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
    



def log_transcript(o, start, now=None, timestamped_file = None):
    if o[0] is None:
        return
    if now is None:
        now = time.time() - start
    
    log_string = f"{now:7.3f}, {o[0]:7.3f}-{o[1]:7.3f} ({(now-o[1]):+2.1f}s): {o[2]}"
    logger.info(log_string)
    if timestamped_file is not None:
        timestamped_file.write(log_string + "\n")
        timestamped_file.flush()
        