

import logging
from threading import Lock
from pathlib import Path
import numpy as np
import time


class Monitor:
    _instance = None
    _lock = Lock()  # ensure thread safety

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Monitor, cls).__new__(cls)
                cls._instance._init_monitor()
            return cls._instance

    def _init_monitor(self):
        self.logger = logging.getLogger("Monitor")
        self.logger.setLevel(logging.DEBUG)

        # File handler for monitor logs.
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler("logs/monitor.log")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.start_time = np.nan

        self.data = []


    def set_start_time(self, start_time):
        self.start_time = start_time
        self.log("General", None, "start_time", start_time, "Recording started.")


    def log(self, category, subcategory, number_name, number, message):
        # Build the log message with timestamp, category, subcategory, metric name and value, and a message.

        self.data.append((category, subcategory, number_name, number, message))

        log_line = f"{category}"
        if subcategory:
            log_line += f".{subcategory}"
        log_line += f" | {number_name}={number} | {message}"
        self.logger.debug(log_line)


    def log_delay(self, category, subcategory, recording_time, message):

        expected = self.start_time + recording_time
        delay = time.time() - expected
        
        self.log(category, subcategory, "delay", delay, message)
     