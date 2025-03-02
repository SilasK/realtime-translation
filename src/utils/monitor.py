import logging
from threading import Lock
from pathlib import Path
import numpy as np
import time
import pandas as pd
import threading


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
        log_file = Path("logs/monitor.log")
        log_file.unlink(missing_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.data_file = Path("logs/monitor_data.csv")
        self.data_file.unlink(missing_ok=True)

        self.start_time = np.nan

        self.data_list = []

        self.data = None
        self.should_monitor = False

    def start(self, start_time=None):
        self.should_monitor = True
        self.start_time = start_time
        self.log("General", None, "start_time", start_time, "Recording started.")

        self.logger.debug("Monitor started.")

        self.analysis_thread = threading.Thread(target=self._analysis_thread)
        self.analysis_thread.start()

    def stop(self):
        self.should_monitor = False
        self.analysis_thread.join()

    def log(self, category, subcategory, number_name, number, message):
        # Build the log message with timestamp, category, subcategory, metric name and value, and a message.

        self.data_list.append(
            (time.time(), category, subcategory, number_name, number, message)
        )

        log_line = f"{category}"
        if subcategory:
            log_line += f".{subcategory}"
        log_line += f" | {number_name}={number} | {message}"

        self.logger.debug(log_line)

    def log_delay(self, category, subcategory, recording_time, message):

        expected = self.start_time + recording_time
        delay = time.time() - expected

        self.log(category, subcategory, "delay", delay, message)

    def collect_data(self):

        if len(self.data_list) > 0:
            new_data = pd.DataFrame(
                self.data_list,
                columns=[
                    "timestamp",
                    "category",
                    "subcategory",
                    "number_name",
                    "number",
                    "message",
                ],
            )
            new_data.timestamp = pd.to_datetime(new_data["timestamp"], unit="s")
            new_data = new_data.set_index("timestamp")

            self.data_list = []

            if self.data is None:
                self.data = new_data
                new_data.to_csv(self.data_file, mode="w", header=True)
            else:
                self.data = pd.concat([self.data, new_data])
                new_data.to_csv(self.data_file, mode="a", header=False)

        return self.data

    def _analysis_thread(self):

        self.logger.debug("Monitor analysis thread started.")
        time.sleep(5)
        while self.should_monitor:
            time.sleep(5)
            # Perform analysis on the data collected.

            if len(self.data_list) > 0:
                self.collect_data()

                delay_data = self.data.query("number_name=='delay'")

                for group, df in delay_data.groupby(
                    ["category", "subcategory", "number_name"]
                ):

                    group_name = group[0]
                    if group[1]:
                        group_name += f".{group[1]}"

                    df = df.iloc[-10:]

                    number_name = group[2]

                    tendance = df.number.diff().dropna()
                    self.logger.info(
                        f"\n Monitor for: {group_name}:{number_name}\n"
                        f"Mean: {df.number.mean()}\n"
                        f"Max: {df.number.max()} - Min: {df.number.min()}\n"
                        f"Tendance: {tendance.mean()}"
                    )
