#!/usr/bin/env python3
import sys
import numpy as np
import time
import logging
import threading
import argparse
import yaml
from pathlib import Path

from src.translation.audio import load_audio, play_audio, load_audio_chunk
from src.translation.translation import TranslationPipeline
from src.whisper_streaming.whisper_online import asr_factory, set_logging

SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)

def load_config(config_file: str) -> argparse.Namespace:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)




def output_transcript(o, start, translation_pipeline, now=None):
    if now is None:
        now = time.time() - start
    if o[0] is not None:
        log_string = f"{now*1000:1.0f}, {o[0]*1000:1.0f}-{o[1]*1000:1.0f} ({(now-o[1]):+1.0f}s): {o[2]}"
        logger.debug(log_string)
        translation_pipeline.put_text(o[2])

def main():
    config_file = "translation_server_config.yaml"
    args = load_config(config_file)
    set_logging(args, logger, others=["src.whisper_streaming.online_asr", "src.translation.translation", "src.translation.audio"])
    if args.input_audio is None or args.input_audio.lower() == "mic":
        raise NotImplementedError("Microphone input is not implemented yet")
    else:
        audio_path = Path(args.input_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file {audio_path} does not exist")
    


        duration = len(load_audio(audio_path)) / SAMPLING_RATE
        logger.info("Audio duration is: %2.2f seconds", duration)

    # initialize_asr
    asr, online = asr_factory(args, logfile=None)
    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size
    

    # initialize_translation_pipeline
    output_folder = Path("translations")
    output_folder.mkdir(exist_ok=True, parents=True)
    translation_pipeline = TranslationPipeline(args.lan, args.target_languages, output_folder=output_folder
    )
    translation_pipeline.start()



    a = load_audio_chunk(audio_path, 0, 1)
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg
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
        else:
            output_transcript(o, start, translation_pipeline)
        
        now = time.time() - start
        logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

        if end >= duration:
            break

    o = online.finish()
    output_transcript(o, start, translation_pipeline, now=None)

if __name__ == "__main__":
    main()