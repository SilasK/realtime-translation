
import signal
from src.translation import server

CONFIG_FILE = "translation_server_config.yaml"

from src.utils.logging import add_other_loggers,logger
import sys

## Set up logging
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time:%H:%M:%S.%f}<level>[{level}]: {message}</level>", colorize=True)

logger.add("logs/translation_server.log", level="DEBUG", format="{time:%H:%M:%S.%f}[{level}]: {message}", colorize=True)

add_other_loggers(["src.whisper_streaming.online_asr", "sr.translation.translation", "src.whisper.audio","src.whisper.timestamped_words"], level="DEBUG")

signal.signal(signal.SIGINT, server.signal_handler)

translation_server_arguments = server.init(CONFIG_FILE,log_to_console=True)

server.main_loop(*translation_server_arguments)