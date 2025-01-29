#!/usr/bin/env python3
import sys

import time
import logging
import threading

from pathlib import Path

SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)

from src.translation.audio import load_audio, play_audio, load_audio_chunk
from src.translation.translation import TranslationPipeline
from src.whisper_streaming.whisper_online import *




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path",
        type=str,
        default='samples_jfk.wav',
        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.",
    )
    add_shared_args(parser)
    parser.add_argument(
        "--start_at",
        type=float,
        default=0.0,
        help="Start processing audio at this time.",
    )
    parser.add_argument(
        "--offline", action="store_true", default=False, help="Offline mode."
    )
    parser.add_argument(
        "--comp_unaware",
        action="store_true",
        default=False,
        help="Computationally unaware simulation.",
    )

    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = None # sys.stderr

    if args.offline and args.comp_unaware:
        logger.error(
            "No or one option from --offline and --comp_unaware are available, not both. Exiting."
        )
        sys.exit(1)


    set_logging(args, logger,others=["src.whisper_streaming.online_asr","src.translation.translation","src.translation.audio"])

    audio_path = args.audio_path


    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)



    asr, online = asr_factory(args, logfile=logfile)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size


## Load translater

    translation_output_folder= Path("translations")
    translation_output_folder.mkdir(exist_ok=True, parents=True)

    translation_pipeline = TranslationPipeline("fr",["en","uk","de"],
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

            if logfile is not None:
                print(
                    log_string,
                    file=logfile,
                    flush=True,
                )

            translation_pipeline.put_text(o[2])
        else:
            # No text, so no output
            pass

    if args.offline:  ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else:  # online = simultaneous mode

        audio_thread = threading.Thread(target=play_audio,args=(audio_path,beg),daemon=True)
        audio_thread.start()

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
