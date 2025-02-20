import signal
from src.translation import server

CONFIG_FILE = "translation_server_config.yaml"

from src.utils.logging import add_other_loggers, logger
from src.utils.monitor import Monitor
from src.utils.logging import InterceptHandler
import sys
from pathlib import Path

## Set up logging
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format="{time:%H:%M:%S.%f}<level>[{level}]: {message}</level>",
    colorize=True,
)

logfile = Path("logs/translation_test.log")
logfile.unlink(missing_ok=True)
logger.add(
    logfile,
    level="DEBUG",
    format="{time:%H:%M:%S.%f}[{level}]: {message}",
    colorize=True,
)


add_other_loggers(
    [
        "src.translation.server",
        "src.whisper_streaming.online_asr",
        "src.translation.translation",
        "src.whisper.audio",
        "src.whisper.timestamped_words",
    ],
    level="DEBUG",
)


signal.signal(signal.SIGINT, server.signal_handler)


args = server.load_config(CONFIG_FILE)


translation_server_arguments = server.initialize(
    args, log_to_console=True, log_to_web=False
)

server.main_loop(args, *translation_server_arguments)
