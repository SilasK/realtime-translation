
import numpy as np
import librosa
from functools import lru_cache

import logging

import sounddevice as sd



SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)



@lru_cache(10**6)
def load_audio(fname, target_sr=SAMPLING_RATE):
    """
    Load an audio file and resample it to the target sampling rate.

    Args:
        fname (str): Path to the audio file.
        target_sr (int): Target sampling rate.

    Returns:
        np.ndarray: Audio data.
    """
    audio, sr_native = librosa.load(fname, sr=None, dtype=np.float32)
    if sr_native != target_sr:
        logger.debug(f"Resampling from {sr_native} Hz to {target_sr} Hz.")
        audio = librosa.resample(audio, orig_sr=sr_native, target_sr=target_sr)
    return audio

def play_audio(audio_path, beg=0):
    """
    Play audio from the specified path starting at the given time.

    Args:
        audio_path (str): Path to the audio file.
        beg (float): Start time in seconds.

    Example:
        >>> play_audio('path/to/audio.wav', 5)
    """
    try:
        audio = load_audio(audio_path)
        
        start_sample = int(beg * SAMPLING_RATE)
        if start_sample >= len(audio):
            logger.error("Start time exceeds audio length.")
            return
        logger.debug(f"Playing audio from {audio_path} starting at {beg} seconds.")
        
        
        sd.play(audio[start_sample:], SAMPLING_RATE)
        sd.wait()  # Wait until playback is finished
    except Exception as e:
        logger.error(f"Error playing audio: {e}")


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * SAMPLING_RATE)
    end_s = int(end * SAMPLING_RATE)
    return audio[beg_s:end_s]
