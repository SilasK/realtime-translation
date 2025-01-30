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


from src.translation.audio import load_audio, play_audio, load_audio_chunk
from src.translation.audio import FileAudioSource, MicrophoneAudioSource

from src.translation.translation import TranslationPipeline
from src.whisper_streaming.whisper_online import asr_factory, set_logging

SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)

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



def log_transcript(o, start, now=None):
    if now is None:
        now = time.time() - start
    if o[0] is not None:
        log_string = f"{now*1000:1.0f}, {o[0]*1000:1.0f}-{o[1]*1000:1.0f} ({(now-o[1]):+1.0f}s): {o[2]}"
        logger.debug(log_string)
        


def process_audio(audio_source, transcriber,translation_pipeline,min_chunk):
    running = True
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
            chunk = audio_source.get_chunk(min_chunk)
            if chunk is None:
                break
                
            transcriber.insert_audio_chunk(chunk)
            try:
                o = transcriber.process_iter()
                log_transcript(o, start)
                translation_pipeline.put_text(o[2])
            except AssertionError as e:
                logger.error(f"Assertion error: {e}")
            
            now = time.time() - start
            logger.debug(f"Processed chunk at {now:.2f}s")
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        audio_source.stop()
        o = transcriber.finish()
        log_transcript(o, start)
        translation_pipeline.put_text(o[2])
        translation_pipeline.stop()


def main():
    config_file = "translation_server_config.yaml"

    args = load_config(config_file)
    set_logging(args, logger, others=["src.whisper_streaming.online_asr", "src.translation.translation", "src.translation.audio"])


    if args.input_audio is None or args.input_audio.lower() == "mic":
        audio_source = MicrophoneAudioSource()
    else:
        audio_source = FileAudioSource(args.input_audio)
    

    # initialize_asr
    asr, transcriber = asr_factory(args, logfile=None)
    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size
    warmup_asr(asr, args.warmup_file)
    
    

    # initialize_translation_pipeline
    output_folder = Path("translations")
    output_folder.mkdir(exist_ok=True, parents=True)
    translation_pipeline = TranslationPipeline(args.lan, args.target_languages, output_folder=output_folder
    )
    translation_pipeline.start()



    process_audio(audio_source, transcriber,translation_pipeline,min_chunk)







if __name__ == "__main__":
    main()