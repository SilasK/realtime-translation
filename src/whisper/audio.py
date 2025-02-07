import numpy as np
import librosa
from functools import lru_cache

import logging
from pathlib import Path
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


import numpy as np
import sounddevice as sd
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class AudioInput:
    def __init__(
        self,
        callback: Callable,
        source: str = "mic",
        sample_rate: int = 16000,
        chunk_size: float = 1.0,
        channels: int = 1,
        dtype: np.dtype = np.float32,
    ):
        """
        Args:
            callback: Function to call with new audio data
            source: "mic" or path to audio file
            sample_rate: Sample rate in Hz
            chunk_size: Size of audio chunks in seconds
            channels: Number of audio channels
            dtype: Numpy dtype for audio data
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.running = False
        
        # Calculate block size in samples
        self.block_size = int(chunk_size * sample_rate)
        
        if source.lower() == "mic":
            self._setup_microphone()
        else:
            self._setup_file_simulation(source)

    def _setup_microphone(self):
        """Configure microphone input"""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.block_size,
            dtype=self.dtype
        )
        
    def _setup_file_simulation(self, file_path: str):
        """Configure file simulation"""
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.audio_data = load_audio(file_path)
        self.position = 0
        self.simulation_thread = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for both real and simulated audio"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Ensure consistent shape
        data = indata.copy()
        if len(data.shape) > 1:
            data = data.flatten()
            
        # Call user callback
        self.callback(data)

    def _simulate_stream(self):
        """Simulate real-time audio from file"""
        chunk_duration = self.chunk_size
        
        while self.running and self.position < len(self.audio_data):
            start_pos = self.position
            end_pos = start_pos + self.block_size
            
            # Get chunk and handle end of file
            chunk = self.audio_data[start_pos:end_pos]
            if len(chunk) < self.block_size:
                chunk = np.pad(chunk, (0, self.block_size - len(chunk)))
            
            # Reshape for callback
            chunk = chunk.reshape(-1, 1)
            
            # Call callback
            self._audio_callback(chunk, self.block_size, None, None)
            
            # Update position
            self.position += self.block_size
            
            # Sleep to simulate real-time
            time.sleep(chunk_duration)

    def start(self):
        """Start audio input"""
        self.running = True
        
        if hasattr(self, 'stream'):
            self.stream.start()
        else:
            self.simulation_thread = threading.Thread(
                target=self._simulate_stream,
                daemon=True
            )
            self.simulation_thread.start()

    def stop(self):
        """Stop audio input"""
        self.running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        elif self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)

def test_audio_input(n_terations=4):
    import tempfile
    import soundfile as sf
    """Test both microphone and file input"""

    

    def audio_callback(data):
        
        current_time = time.time()
        time_diff = current_time - audio_callback.last_time
        audio_callback.last_time = current_time
        
        logger.info(f"Got audio chunk: shape={data.shape}, iteration={audio_callback.iteration}, time_diff={time_diff:.4f}s")
        
        if audio_callback.iteration <=1:
            
            sleep_time = 0.5
            logger.info(f"Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        audio_callback.iteration += 1


    audio_callback.iteration = 1
    audio_callback.last_time = np.nan
        

    # Test microphone
    logger.info("Testing microphone input...")
    mic_input = AudioInput(audio_callback, source="mic")
    mic_input.start()
    time.sleep(n_terations)
    mic_input.stop()


    audio_callback.iteration = 1
    audio_callback.last_time = np.nan

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        # Record 5 seconds of audio from the microphone and save to a file
        temp_audio_path = temp_audio_file.name

        logger.info("Recording 5 seconds of audio...")
        recording = sd.rec(int(5 * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished


        sf.write(temp_audio_path, recording.flatten(), SAMPLING_RATE )
        logger.debug(f"Recorded audio saved to {temp_audio_path}")


        logger.info("Testing file input...")

        file_input = AudioInput(
            audio_callback,
            source=temp_audio_path
        )
        file_input.start()
        time.sleep(n_terations)
        file_input.stop()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    test_audio_input()



