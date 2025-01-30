
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




from abc import ABC, abstractmethod
import queue
import sounddevice as sd
import time


class AudioSource(ABC):
    def __init__(self,chunk_size=1.0,sample_rate=SAMPLING_RATE):
        """Initialize an audio source.
        
        Args:
            chunk_size (float): Size of audio chunks in seconds
            sample_rate (int): Sample rate of the audio source
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
    
    @abstractmethod
    def start(self): pass
    
    @abstractmethod
    def stop(self): pass
    
    @abstractmethod
    def get_chunk(self, chunk_size): pass

class FileAudioSource(AudioSource):
    """
    A class representing an audio source from a file.
    This class inherits from AudioSource and provides functionality to stream audio data
    from a file in chunks.
    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file to be loaded
    start_from : float, optional
        Starting position in seconds to begin streaming from (default is 0)
    chunk_size : float
        Size of audio chunks in seconds to be yielded
    sample_rate : int, optional
        Sampling rate of the audio in Hz (default is SAMPLING_RATE)
    Attributes
    ----------
    audio_path : Path
        Path object representing the location of the audio file
    audio_data : numpy.ndarray
        Loaded audio data
    position : float
        Current position in the audio stream in seconds
    running : bool
        Flag indicating if the audio source is currently streaming
    start_time : float or None
        Timestamp when streaming started, None if not started
    Notes
    -----
    The audio file is loaded entirely into memory upon initialization.
    """

    def __init__(self, audio_path,  chunk_size:float,
                    start_from:float=0, sample_rate=SAMPLING_RATE):


        super().__init__(chunk_size, sample_rate)

        self.audio_path = Path(audio_path)
        self.audio_data = load_audio(audio_path)
        self.position = start_from
        self.running = False
        self.start_time = None
        duration = len(self.audio_data) / sample_rate
        logger.debug(f"Loaded audio from {audio_path} with duration {duration:.2f}s.")
    

        self.position = 0
        self.duration = len(self.audio_data) / sample_rate
    
    class FileAudioSource(AudioSource):
        def __init__(self, audio_path, chunk_size:float,start_from:float=0, sample_rate=SAMPLING_RATE):
            """Initialize an AudioStream object.
            Args:
                audio_path: Path to the audio file to stream
                start_from (float): Position in seconds to start streaming from (default: 0)
                sample_rate (int): Sample rate of the audio
                chunk_size (float): Size of audio chunks in seconds (default: 1.0)
            """
            self.audio_path = Path(audio_path)
            self.sample_rate = sample_rate
            self.audio_data = load_audio(audio_path)
            self.position = start_from
            self.chunk_size = chunk_size
            self.running = False
            self.start_time = None
            duration = len(self.audio_data) / sample_rate
            logger.debug(f"Loaded audio from {audio_path} with duration {duration:.2f}s.")
        
    def start(self):
        self.start_time = time.time() - self.position
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_chunk(self):
        """Get a chunk of audio data, simulating real-time capture."""
        if not self.running:
            logger.debug("Audio source is not running.")
            return None
            
        current_time = time.time()
        expected_position = current_time - self.start_time
        
        if expected_position < self.position:
            logger.debug("You cannot access audio data from the future.")
            return None
            
        if self.position >= len(self.audio_data) / self.sample_rate:
            logger.debug("End of audio file.")
            return None
            
        start_pos = int(self.position * self.sample_rate)
        end_pos = int((self.position + self.chunk_size) * self.sample_rate)
        chunk = self.audio_data[start_pos:end_pos]
        self.position += self.chunk_size
        return chunk







class MicrophoneAudioSource(AudioSource):
    def __init__(self, chunk_size:float,sample_rate=SAMPLING_RATE):
        super().__init__(chunk_size, sample_rate)


        self.audio_queue = queue.Queue()
        self.running = False
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            dtype=np.float32
        )
        self.stream.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def get_chunk(self):
        if not self.running:
            return None
        try:
            
            chunk = self.audio_queue.get(timeout=self.chunk_size)
            return chunk.flatten()
        except queue.Empty:
            return np.zeros(int(self.chunk_size * self.sample_rate))


def test_audio_source(get_audio_source,n_chunks=5, chunk_size=1.0):

    logger.info("Ask for audio with normal speed")
    audio_source = get_audio_source()
    audio_source.start()

    for i in range(n_chunks):
        time.sleep(chunk_size)
        chunk = audio_source.get_chunk()
        print(f"Got chunk {i} with length {len(chunk)}")

        import pdb; pdb.set_trace()
        
    audio_source.stop()

    logger.info("Ask for audio too fast")
    audio_source = get_audio_source()
    audio_source.start()

    for i in range(n_chunks):
        time.sleep(0.5*chunk_size)
        chunk = audio_source.get_chunk()
        if chunk is None:
            print("Got no audio. asking too fast")
        else:
            print(f"Got chunk {i} with length {len(chunk)}")

    audio_source.stop()

    logger.info("Ask for audio too slow")
    audio_source = get_audio_source()
    audio_source.start()

    for i in range(n_chunks):

        time.sleep(1.5*chunk_size)
        chunk = audio_source.get_chunk()
        print(f"Got chunk {i} with length {len(chunk)}")

    audio_source.stop()

    logger.info("Ask for audio after stopping")
    audio_source = get_audio_source()
    audio_source.start()
    time.sleep(0.5*chunk_size)
    audio_source.stop()

    for i in range(n_chunks):
        chunk = audio_source.get_chunk()
        if chunk is None:
            print("Got no audio. asking after stopping")
        else:
            print(f"Got chunk {i} with length {len(chunk)}")

    







if __name__ == "__main__":
    # First record some audio and save to temp file
    import tempfile
    import os
    import time
    import soundfile as sf
    logging.basicConfig(level=logging.DEBUG)


    logger.info("Testing microphone audio source")

    def get_audio_source():
        return MicrophoneAudioSource(chunk_size=1.0)
    
    test_audio_source(get_audio_source)



    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_file = temp_audio_file.name

        audio_source = MicrophoneAudioSource(chunk_size=1.0)
        audio_source.start()
        time.sleep(5)
        audio_source.stop()

        audio_data = audio_source.audio_queue.queue
        audio_data = np.concatenate(list(audio_data))
        sf.write(temp_file, audio_data, SAMPLING_RATE )

        # load the audio into the audio source

        logger.info("Testing file audio source")

        def get_audio_source():
            return FileAudioSource(temp_file, chunk_size=1.0)
        
        test_audio_source(get_audio_source)

    # Now test the microphone audio source


 


    
