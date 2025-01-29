#!/usr/bin/env python3
import sys
import numpy as np
import time
import logging
import threading
import argparse
import yaml

from pathlib import Path

SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)





from src.translation.audio import load_audio, play_audio, load_audio_chunk
from src.translation.translation import TranslationPipeline
from src.whisper_streaming.whisper_online import *




if __name__ == "__main__":



    config = yaml.safe_load(open("translation_server_config.yaml"))
    args = argparse.Namespace(**config)

    set_logging(args, logger,others=["src.whisper_streaming.online_asr","src.translation.translation","src.translation.audio"])

    # Get audio path or mic
    if config["input_audio"] is None or config["input_audio"].lower() == "mic":

        raise NotImplementedError("Microphone input is not implemented yet")
    else:
        audio_path = config["input_audio"]
        assert Path(audio_path).exists(), f"Audio file {audio_path} does not exist"

        duration = len(load_audio(audio_path)) / SAMPLING_RATE
        logger.info("Audio duration is: %2.2f seconds" % duration)
        



    asr, online = asr_factory(args, logfile=None)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size


## Load translater

    translation_output_folder= Path("translations")
    translation_output_folder.mkdir(exist_ok=True, parents=True)

    translation_pipeline = TranslationPipeline(args.lan, args.target_languages,
                                               output_folder=translation_output_folder
                                   )
    translation_pipeline.start()



    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            log_string = f"{now*1000:1.0f}, {o[0]*1000:1.0f}-{o[1]*1000:1.0f} ({(now-o[1]):+1.0f}s): {o[2]}"

            logger.debug(
                log_string
            )



            translation_pipeline.put_text(o[2])
        else:
            # No text, so no output
            pass






    end = 0
    while True:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        end = time.time() - start
        a = load_audio_chunk(audio_path, beg, end)
        beg = end
        online.insert_audio_chunk(a)

        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {e}")
            pass
        else:
            output_transcript(o)
        now = time.time() - start
        logger.debug(
            f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}"
        )

        if end >= duration:
            break
    now = None

    o = online.finish()
    output_transcript(o, now=now)
