import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
from timed_objects import ASRToken, Sentence, Transcript, TimedList

logger = logging.getLogger(__name__)


class HypothesisBuffer:
    """
    Buffer to store and process ASR hypothesis tokens.

    It holds:
      - committed_in_buffer: tokens that have been confirmed (committed)
      - buffer: the last hypothesis that is not yet committed
      - new: new tokens coming from the recognizer
    """
    def __init__(self, sep: str, logfile=sys.stderr, confidence_validation=False):
        self.sep = sep
        self.confidence_validation = confidence_validation
        self.committed_in_buffer: TimedList = TimedList([],sep=sep) 
        self.buffer: TimedList = TimedList([],sep=sep)
        self.new: TimedList = TimedList([],sep=sep)
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None
        self.logfile = logfile
        

    def insert(self, new_tokens: TimedList, offset: float):
        """
        Insert new tokens (after applying a time offset) and compare them with the 
        already committed tokens. Only tokens that extend the committed hypothesis 
        are added.
        """
        # Apply the offset to each token.
        new_tokens.shift(offset)
        if self.sep != new_tokens.sep:
            logger.warning(f"Separator mismatch: {self.sep} != {new_tokens.sep}")
        # Only keep tokens that are roughly “new”
        self.new = TimedList([token for token in new_tokens if token.start > self.last_committed_time - 0.1],sep= self.sep)

        if self.new:
            first_token = self.new[0]
            if abs(first_token.start - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    committed_len = len(self.committed_in_buffer)
                    new_len = len(self.new)
                    # Try to match 1 to 5 consecutive tokens
                    max_ngram = min(min(committed_len, new_len), 5)
                    for i in range(1, max_ngram + 1):
                        committed_ngram = self.committed_in_buffer[-i:].get_text(sep=self.sep)
                        new_ngram = self.new[:i].get_text(sep=self.sep)
                        if committed_ngram == new_ngram:
                            removed = self.new[:i].get_text(sep=self.sep)
                            self.new = self.new[i:]
                            logger.debug(f"Removing last {i} words: {removed}")
                            break

    def flush(self) -> TimedList:
        """
        Returns the committed chunk, defined as the longest common prefix
        between the previous hypothesis and the new tokens.
        """
        committed: TimedList = TimedList([],sep=self.sep)
        while self.new:
            current_new = self.new[0]
            if self.confidence_validation and current_new.probability and current_new.probability > 0.95:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.new.pop(0)
                self.buffer.pop(0) if self.buffer else None
            elif not self.buffer:
                break
            elif current_new.text == self.buffer[0].text:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = TimedList([],sep=self.sep)
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        """
        Remove tokens (from the beginning) that have ended before `time`.
        """
        _,self.committed_in_buffer= self.committed_in_buffer.split_at(time, edge_word_goes_left=False)



class OnlineASRProcessor:
    """
    Processes incoming audio in a streaming fashion, calling the ASR system
    periodically, and uses a hypothesis buffer to commit and trim recognized text.
    
    The processor supports two types of buffer trimming:
      - "sentence": trims at sentence boundaries (using a sentence tokenizer)
      - "segment": trims at fixed segment durations.
    """
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):
        """
        asr: An ASR system object (for example, a WhisperASR instance) that
             provides a `transcribe` method, a `ts_words` method (to extract tokens),
             a `segments_end_ts` method, and a separator attribute `sep`.
        tokenize_method: A function that receives text and returns a list of sentence strings.
        buffer_trimming: A tuple (option, seconds), where option is either "sentence" or "segment".
        """
        self.asr = asr
        self.tokenize = tokenize_method
        self.logfile = logfile
        self.confidence_validation = confidence_validation
        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

        if self.buffer_trimming_way not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming must be either 'sentence' or 'segment'")
        if self.buffer_trimming_sec <= 0:
            raise ValueError("buffer_trimming_sec must be positive")
        elif self.buffer_trimming_sec > 30:
            logger.warning(
                f"buffer_trimming_sec is set to {self.buffer_trimming_sec}, which is very long. It may cause OOM."
            )

    def init(self, offset: Optional[float] = None):
        """Initialize or reset the processing buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(sep=self.asr.sep,logfile=self.logfile, confidence_validation=self.confidence_validation)
        self.buffer_time_offset = offset if offset is not None else 0.0
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed: TimedList = TimedList([],sep=self.asr.sep)

    def insert_audio_chunk(self, audio: np.ndarray):
        """Append an audio chunk (a numpy array) to the current audio buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> Tuple[str, str]:
        """
        Returns a tuple: (prompt, context), where:
          - prompt is a 200-character suffix of committed text that falls 
            outside the current audio buffer.
          - context is the committed text within the current audio buffer.
        """
        k = len(self.committed)
        while k > 0 and self.committed[k - 1].end > self.buffer_time_offset:
            k -= 1

        prompt_tokens = self.committed[:k]
        prompt_words = [token.text for token in prompt_tokens]
        prompt_list = []
        length_count = 0
        # Use the last words until reaching 200 characters.
        while prompt_words and length_count < 200:
            word = prompt_words.pop(-1)
            length_count += len(word) + 1
            prompt_list.append(word)
        non_prompt_tokens = self.committed[k:]
        context_text = self.asr.sep.join(token.text for token in non_prompt_tokens)
        return self.asr.sep.join(prompt_list[::-1]), context_text

    def get_buffer(self) -> Transcript:
        """
        Get the unvalidated buffer in TimedText format.
        """
        return self.transcript_buffer.buffer.concatenate(sep=self.asr.sep)
        

    def transcribe_audio_buffer(self) -> TimedList:
        """
        Transcribe the current audio buffer.
        """
        prompt_text, _ = self.prompt()
        logger.debug(
            f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds from {self.buffer_time_offset:.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_text)
        return res

    def process_iter(self) -> TimedList:
        """
        Processes the current audio buffer.

        Returns a Transcript object representing the committed transcript.
        """
        
        res = self.transcribe_audio_buffer()
        tokens = self.asr.ts_words(res)  # Expecting TimedList  

        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        self.committed.extend(committed_tokens)
        logger.debug(f">>>> COMPLETE NOW: {committed_tokens.get_text()}")
        logger.debug(f"INCOMPLETE: {self.transcript_buffer.buffer.get_text()}")

        # If we have completed sentences, chunk the audio buffer
        if committed_tokens and self.buffer_trimming_way == "sentence":
            self.chunk_completed_sentence()

        if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:

            if self.buffer_trimming_way == "sentence":
                logger.warning(f"Chunking after {self.buffer_trimming_sec}s even if no completed sentences!")
            else:
                logger.debug("Chunking segment")

            self.chunk_completed_segment(res)
            
        logger.debug(
            f"Length of audio buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds"
        )
        return committed_tokens

    def chunk_completed_sentence(self):
        """
        If the committed tokens form at least sentences, chunk the audio
        buffer at the end time of the ultimate sentence.
        """

        if not self.committed:
            return
        logger.debug("COMPLETED SENTENCE: " + self.committed.get_text())

        sentences = self.committed.split_to_sentences(self.tokenize)
        for sentence in sentences:
            logger.debug(f"\tSentence: {sentence}")
        if len(sentences) <= 1:
            return
        # Keep the last sentences.
        chunk_time = sentences[-1].end
        logger.debug(f"--- Sentence chunked at {chunk_time:.2f}")
        self.chunk_at(chunk_time)

    def chunk_completed_segment(self, res):
        """
        Chunk the audio buffer based on segment-end timestamps reported by the ASR.
        """
        if not self.committed:
            return
        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.committed[-1].end
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_time:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= last_committed_time:
                logger.debug(f"--- Segment chunked at {e:.2f}")
                self.chunk_at(e)
            else:
                logger.debug("--- Last segment not within committed area")
        else:
            logger.debug("--- Not enough segments to chunk")

    def chunk_at(self, time: float):
        """
        Trim both the hypothesis and audio buffer at the given time.
        """
        logger.debug(f"Chunking at {time:.2f}s")
        logger.debug(
            f"Audio buffer length before chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time
        logger.debug(
            f"Audio buffer length after chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )



    def finish(self) -> TimedList:
        """
        Flush the remaining transcript when processing ends.
        """
        res = self.transcribe_audio_buffer()
        tokens = self.asr.ts_words(res)
        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        # self.committed.extend(committed_tokens)  
        logger.debug(f">>>> COMPLETE NOW: {committed_tokens.get_text()}")
        
        remaining_tokens = self.transcript_buffer.buffer
        # self.committed.extend(remaining_tokens)

        logger.debug(f"Final non-committed transcript: {remaining_tokens}")

        

        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self.committed= TimedList([],sep=self.asr.sep)

        return committed_tokens+remaining_tokens



class VACOnlineASRProcessor:
    """
    Wraps an OnlineASRProcessor with a Voice Activity Controller (VAC).
    
    It receives small chunks of audio, applies VAD (e.g. with Silero),
    and when the system detects a pause in speech (or end of an utterance)
    it finalizes the utterance immediately.
    """
    SAMPLING_RATE = 16000

    def __init__(self, online_chunk_size: float, *args, **kwargs):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(*args, **kwargs)

        # Load a VAD model (e.g. Silero VAD)
        import torch
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        from silero_vad_iterator import FixedVADIterator

        self.vac = FixedVADIterator(model)
        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status: Optional[str] = None  # "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray):
        """
        Process an incoming small audio chunk:
          - run VAD on the chunk,
          - decide whether to send the audio to the online ASR processor immediately,
          - and/or to mark the current utterance as finished.
        """
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            # VAD returned a result; adjust the frame number
            frame = list(res.values())[0] - self.buffer_offset
            if "start" in res and "end" not in res:
                self.status = "voice"
                send_audio = self.audio_buffer[frame:]

                if len(self.online.audio_buffer) > 0:
                    logger.critical(f"Re-init translator, when previous audio_buffer is not finished!")
                
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif "end" in res and "start" not in res:
                self.status = "nonvoice"
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = "nonvoice"
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == "voice":
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # Keep 1 second worth of audio in case VAD later detects voice,
                # but trim to avoid unbounded memory usage.
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self) -> TimedList:
        """
        Depending on the VAD status and the amount of accumulated audio,
        process the current audio chunk.
        """
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            return self.online.process_iter()
        else:
            logger.debug("No online update, only VAD")
            return TimedList([],sep=self.online.asr.sep)

    def finish(self) -> TimedList:
        """Finish processing by flushing any remaining text."""
        result = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return result
    
    def get_buffer(self) -> Transcript:
        """
        Get the unvalidated buffer in TimedText format.
        """
        return self.online.transcript_buffer.buffer.concatenate(sep=self.online.asr.sep)
