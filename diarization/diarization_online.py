import asyncio
import re
import threading
import numpy as np
import logging


from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sources import AudioSource
from timed_objects import SpeakerSegment
from diart.sources import MicrophoneAudioSource
from rx.core import Observer
from typing import Tuple, Any, List
from pyannote.core import Annotation

logger = logging.getLogger(__name__)

def extract_number(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None

class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""
    
    def __init__(self):
        self.speaker_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()
    
    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value
        
        logger.debug("\n--- New Diarization Result ---")
        
        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        logger.debug(f"Audio shape: {audio.data.shape}")
        
        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end            
            if annotation and len(annotation._labels) > 0:
                logger.debug("\nSpeaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        print(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.speaker_segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=start,
                            end=end
                        ))
            else:
                logger.debug("\nNo speakers detected in this segment")
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
    
    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")
        
    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Custom AudioSource that blocks in read() until close() is called.
    Use push_audio() to inject PCM chunks.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000):
        super().__init__(uri, sample_rate)
        self._closed = False
        self._close_event = threading.Event()

    def read(self):
        self._close_event.wait()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stream.on_completed()
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        if not self._closed:
            new_audio = np.expand_dims(chunk, axis=0)
            logger.debug('Add new chunk with shape:', new_audio.shape)
            self.stream.on_next(new_audio)


class DiartDiarization:
    def __init__(self, sample_rate: int, use_microphone: bool = False):
        self.pipeline = SpeakerDiarization()        
        self.observer = DiarizationObserver()
        
        if use_microphone:
            self.source = MicrophoneAudioSource()
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(uri="websocket_source", sample_rate=sample_rate)
            self.source = self.custom_source
            
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)            
        self.observer.clear_old_segments()        
        return self.observer.get_segments()

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

    def assign_speakers_to_tokens(self, end_attributed_speaker, tokens: list) -> float:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        Uses the segments collected by the observer.
        """
        segments = self.observer.get_segments()
        
        for token in tokens:
            for segment in segments:
                if not (segment.end <= token.start or segment.start >= token.end):
                    token.speaker = extract_number(segment.speaker) + 1
                    end_attributed_speaker = max(token.end, end_attributed_speaker)
        return end_attributed_speaker