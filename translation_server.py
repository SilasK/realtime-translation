#!/usr/bin/env python3
import sys
import numpy as np
import time
import logging
import threading
import argparse
import yaml
from pathlib import Path
import signal

import logging


from src.translation.logging import add_other_loggers,logger

logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time:%H:%M:%S.%f}<level>[{level}]: {message}</level>", colorize=True)

logger.add("logs/translation_server.log", level="DEBUG", format="{time:%H:%M:%S.%f}[{level}]: {message}", colorize=True)








from src.translation.audio import load_audio, play_audio, load_audio_chunk
from src.translation.audio import AudioInput

from src.translation.translation import TranslationPipeline
from src.whisper_streaming.whisper_online import asr_factory, set_logging

SAMPLING_RATE = 16000


def load_config(config_file: str) -> argparse.Namespace:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def warmup_asr(asr, warmup_file):
    if warmup_file is not None:
        logger.info(f"Warming up ASR with {warmup_file}")
        audio = load_audio(warmup_file)
        asr.transcribe(audio)
        logger.info("ASR warmed up.")



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
        


def main():
    config_file = "translation_server_config.yaml"

    args = load_config(config_file)
    add_other_loggers(["src.whisper_streaming.online_asr", "src.translation.translation", "src.translation.audio"], level="DEBUG")



    # initialize_asr
    asr, transcriber = asr_factory(args, logfile=None)
    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size
    warmup_asr(asr, args.warmup_file)

    def put_audiochunk_in_transcriber(chunk):
        transcriber.insert_audio_chunk(chunk)
        

    audio_source = AudioInput(callback= put_audiochunk_in_transcriber, source=args.input_audio, sample_rate=SAMPLING_RATE, chunk_size=min_chunk)



    # initialize_translation_pipeline
    output_folder = Path("translations")
    output_folder.mkdir(exist_ok=True, parents=True)
    translation_pipeline = TranslationPipeline(args.lan, args.target_languages, output_folder=output_folder
    )
    translation_pipeline.start()

    timestamped_file = open(output_folder / f"transcript_with_timestamps.txt","w")




    #def process_audio(audio_source, transcriber,translation_pipeline,min_chunk):
    running = True

    ## Keyboard interrupt handler
    def signal_handler(signum, frame):
        nonlocal running
        logger.info("Stopping audio processing...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)




    
    try:


        logger.info("Ready to process audio.")
        audio_source.start()
        

        
        start = time.time()
        while running:
            try:
                o,incomplete = transcriber.process_iter()
                if o[0] is  None:
                    if not args.vac:
                        logger.warning("No output from transcriber.")
                    time.sleep(0.25*min_chunk)
                    continue
                else:
                    log_transcript(o, start,timestamped_file=timestamped_file)
                    translation_pipeline.put_text(o[2])
            except Exception as e:
                logger.error(f"Assertion error: {e}")
                raise e
            
            now = time.time() - start
            logger.debug(f"Processed chunk at {now:.2f}s")
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        audio_source.stop()
        o,incomplete = transcriber.finish()
        log_transcript(o, start,timestamped_file=timestamped_file)
        timestamped_file.close()
        translation_pipeline.put_text(o[2])
        translation_pipeline.stop()






if __name__ == "__main__":
    main()