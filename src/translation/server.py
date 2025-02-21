from pathlib import Path

import numpy as np

import logging

logger = logging.getLogger(__name__)
import time
import shutil

from pathlib import Path
import queue


from ..whisper.audio import AudioInput

from .translation import TranslationPipeline
from ..whisper_streaming.whisper_online import asr_factory
from ..whisper_streaming.online_asr import words_to_sentences
from ..utils.logging import log_transcript
from ..utils.monitor import Monitor


monitor = Monitor()


SAMPLING_RATE = 16000

import argparse
import yaml


def load_config(config_file: str) -> argparse.Namespace:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def initialize(args, log_to_console=True, log_to_web=False):

    logger.info("Initializing translation pipeline.")

    output_folder = args.output_dir
    if output_folder is not None:

        output_folder = Path(output_folder)
        shutil.rmtree(output_folder, ignore_errors=True)
        output_folder.mkdir(exist_ok=True, parents=True)

    # initialize_asr
    asr, transcriber = asr_factory(args, logfile=None)

    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size
    logger.info(f"Minimum chunk size: {min_chunk}")

    if args.warmup_file:
        _ = asr.transcribe_file(args.warmup_file)
        logger.info("ASR warmup completed.")

    # callback funcion for audio
    def put_audiochunk_in_transcriber(chunk):
        start_time = time.time()
        transcriber.insert_audio_chunk(chunk)
        time_taken = time.time() - start_time

        monitor.log(
            "General",
            "Audio",
            "Time taken to insert audio",
            time_taken,
            "With VAC" if args.vac else "No VAC",
        )

    audio_input_queue = queue.Queue()

    audio_source = AudioInput(
        callback=audio_input_queue.put,
        source=args.input_audio,
        sample_rate=SAMPLING_RATE,
        chunk_size=min_chunk,
    )

    # initialize_translation_pipeline

    translation_pipeline = TranslationPipeline(
        args.lan,
        args.target_languages,
        output_folder=output_folder,
        log_to_console=log_to_console,
        log_to_web=log_to_web,
    )

    # translation_pipeline.start()

    logger.info("Everything set up!")

    return audio_source, transcriber, translation_pipeline, min_chunk, audio_input_queue


### Main loop

translation_loop_running = True


## Keyboard interrupt handler
def signal_handler(signum, frame):
    global translation_loop_running
    logger.info("Stopping audio processing...")
    translation_loop_running = False


def main_loop(
    args, audio_source, transcriber, translation_pipeline, min_chunk, audio_input_queue
):

    try:

        logger.info("Ready to process audio.")
        audio_source.start()

        start = time.time()
        monitor.start(start)

        last_transcribed = np.nan
        while translation_loop_running:
            try:

                # monitor.log("General","Transcription","Time since last transcribed",time.time()- last_transcribed,"This is for debug")

                # insert audio

                audio_queue_size = audio_input_queue.qsize()
                monitor.log(
                    "General",
                    "Audio",
                    "Audio queue size",
                    audio_queue_size,
                    "",
                )
                logger.debug(f"Audio queue size: {audio_queue_size}")

                if audio_queue_size == 0:
                    time.sleep(0.9 * min_chunk)
                    continue

                # get all audio chunks and concatenate them
                audio_chunks = []
                while not audio_input_queue.empty():
                    audio_chunks.append(audio_input_queue.get())

                audio = np.concatenate(audio_chunks)
                transcriber.insert_audio_chunk(audio)

                o, incomplete = transcriber.process_iter()
                last_transcribed = time.time()

                if o[0] is None and incomplete[0] is None:
                    if not args.vac:
                        logger.warning("No output from transcriber.")

                    time.sleep(0.9 * min_chunk)

                if o[0] is not None:

                    translation_pipeline.put_text(o, is_complete=True)

                if incomplete[0] is not None:

                    logger.debug("Incomplete: " + incomplete[2])
                    translation_pipeline.put_text(incomplete, is_complete=False)

            except Exception as e:
                logger.error(f"Error: {e}")
                raise e

            # monitor.log("General", None, "processed audio", time.time() - start, "Time since start")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise e
    finally:
        audio_source.stop()
        transcriber.close()
        translation_pipeline.stop()
        monitor.stop()
