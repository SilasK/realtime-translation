import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        """
        compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        The new tail is added to self.new
        """

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        "Remove (from the beginning) of commited_in_buffer all the words that are finished before `time`"
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer





class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method=None,
        buffer_trimming=("segment", 15),
        logfile=sys.stderr,
    ):
        """
        Initialize OnlineASRProcessor.

        Args:
            asr: WhisperASR object
            tokenize_method: Sentence tokenizer function for the target language.
            Must be a function that takes a list of text as input like MosesSentenceSplitter.
            Can be None if using "segment" buffer trimming option.
            buffer_trimming: Tuple of (option, seconds) where:
            - option: Either "sentence" or "segment"
            - seconds: Number of seconds threshold for buffer trimming
            Default is ("segment", 15)
            logfile: File to store logs

        """
        self.asr = asr
        self.tokenize = tokenize_method
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

        if self.buffer_trimming_way not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming must be either 'sentence' or 'segment'")
        if self.buffer_trimming_sec <= 0:
            raise ValueError("buffer_trimming_sec must be positive")
        elif self.buffer_trimming_sec > 30:
            raise ValueError(
                f"buffer_trimming_sec is set to {self.buffer_trimming_sec}, which is larger than the max for Whisper."
            )

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.len_audio_buffer_last_transcribed= 0
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.final_transcript = []
        self.commited_not_final = []
        
 

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.

        
        """        
        
        if len(self.final_transcript) == 0:
            prompt=""

        if len(self.final_transcript) == 1:
            prompt = self.final_transcript[0][2][-200:]
        
        else:
            prompt = self.concatenate_tsw(self.final_transcript)[2][-200:]
        # TODO: this is not ideal as we concatenate each time the whole transcript

        # k = max(0, len(self.final_transcript) - 1)
        # while k > 1 and self.final_transcript[k - 1][1] > self.buffer_time_offset:
        #     k -= 1

        # p = self.final_transcript[:k]

        
        # p = [t for _, _, t in p]
        # prompt = []
        # l = 0
        # while p and l < 200:  # 200 characters prompt size
        #     x = p.pop(-1)
        #     l += len(x) + 1
        #     prompt.append(x)

        non_prompt =  self.concatenate_tsw(self.commited_not_final)[2]

        logger.debug(f"PROMPT(previous): {prompt[:20]}…{prompt[-20:]} (length={len(prompt)}chars)")
        logger.debug(f"CONTEXT: {non_prompt}")

        return prompt, non_prompt
        

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """

        
        len_audio_buffer_now= len(self.audio_buffer)
        if len_audio_buffer_now == self.len_audio_buffer_last_transcribed:
            logger.debug("No new audio data.")
            return (None, None, "")
    
        self.len_audio_buffer_last_transcribed = len_audio_buffer_now
        logger.info(
            f"transcribing {len_audio_buffer_now/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )

        prompt, non_prompt = self.prompt()

        ## Transcribe and format the result to [(beg,end,"word1"), ...]
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(res)


        # insert into HypothesisBuffer, and get back the commited words
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        commited_tsw = self.transcript_buffer.flush()

        
        
        if len(commited_tsw) == 0:
            return (None, None, "")
        else:
            logger.debug(f"Newly commited words: {self.concatenate_tsw(commited_tsw)[2]}")


        self.commited_not_final.extend(commited_tsw)


        # Define `completed` and `the_rest` based on the buffer_trimming_way
        # completed will be returned at the end of the function.
        # completed is a transcribed text with (beg,end,"sentence ...") format.


        completed = []
        if self.buffer_trimming_way == "sentence":
            
            sentences = self.words_to_sentences(self.commited_not_final)



            if len(sentences) < 2:
                logger.debug(f"[Sentence-segmentation] no full sentence segmented, do not commit anything.")
                
                

            
            else:
                identified_sentence= "\n    - ".join([f"{s[0]*1000:.0f}-{s[1]*1000:.0f} {s[2]}" for s in sentences])
                logger.debug(f"[Sentence-segmentation] identified sentences:\n    - {identified_sentence}")

                # assume last sentence is incomplete, which is not always true

                # we will continue with audio processing at this timestamp
                chunk_at = sentences[-2][1]

                self.chunk_at(chunk_at)
                # TODO: here paragraph breaks can be added
                self.commited_not_final = sentences[-1:]

                completed= sentences[:-1]

            
        


            # break audio buffer anyway if it is too long

        if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec :
                    
            if self.buffer_trimming_way == "sentence":
                logger.warning(f"Chunck segment after {self.buffer_trimming_sec} seconds!"
                                " Even if no sentence was found!"
                            )
                    
                    

                
            completed = self.chunk_completed_segment() 
                

            

        

        if len(completed) == 0:      
            return (None, None, "")
        else:
            self.final_transcript.extend(completed) # add whole time stamped sentences / or words to commited list
        

            completed_text_segment= self.concatenate_tsw(completed)
            
            the_rest = self.concatenate_tsw(self.transcript_buffer.complete())
            commited_but_not_final = self.concatenate_tsw(self.commited_not_final)
            logger.debug(f"\n    COMPLETE NOW: {completed_text_segment[2]}\n"
                         f"    COMMITTED (but not Final): {commited_but_not_final[2]}\n"
                         f"    INCOMPLETE: {the_rest[2]}"
                         )


            return completed_text_segment


    def chunk_completed_segment(self) -> list:

        
        ts_words = self.commited_not_final

        if len(ts_words) <= 1:
            logger.debug(f"--- not enough segments to chunk (<=1 words)")
            return []
        else:

            ends = [w[1] for w in ts_words]

            t = ts_words[-1][1] # start of the last word
            e = ends[-2] 
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] 

            if e <= t:
                
                self.chunk_at(e)

                n_commited_words = len(ends)-1

                words_to_commit = ts_words[:n_commited_words]
                self.final_transcript.extend(words_to_commit)
                self.commited_not_final = ts_words[n_commited_words:]

                return words_to_commit



            else:
                logger.debug(f"--- last segment not within commited area")
                return []


    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        logger.debug(f"chunking at {time:2.2f}s")

        logger.debug(
            f"len of audio buffer before chunking is: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}s"
            )


        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE) :]
        self.buffer_time_offset = time

        logger.debug(
            f"len of audio buffer is now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}s"
            )

    def words_to_sentences(self, words):
        """Uses self.tokenize for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """


        cwords = [w for w in words]
        t = self.asr.sep.join(o[2] for o in cwords)
        logger.debug(f"[Sentence-segmentation] Raw Text: {t}")

        s = self.tokenize([t])
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                if end is None and sent == w:
                    end = e
                if beg is not None and end is not None:
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.concatenate_tsw(o)
        if f[1] is not None:
            logger.debug(f"last, noncommited: {f[0]*1000:.0f}-{f[1]*1000:.0f}: {f[2]}")
        self.buffer_time_offset += len(self.audio_buffer) / 16000
        return f

    def concatenate_tsw(
        self,
        tsw,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep

        

        t = sep.join(s[2] for s in tsw)
        if len(tsw) == 0:
            b = None
            e = None
        else:
            b = offset + tsw[0][0]
            e = offset + tsw[-1][1]
        return (b, e, t)


class VACOnlineASRProcessor(OnlineASRProcessor):
    """Wraps OnlineASRProcessor with VAC (Voice Activity Controller).

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds),
    it runs VAD and continuously detects whether there is speech or not.
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    """

# TODO: VACOnlineASRProcessor does not break after chunch length is reached, this can lead to overflow!

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch

        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        from src.whisper_streaming.silero_vad_iterator import FixedVADIterator

        self.vac = FixedVADIterator(
            model
        )  # we use the default options there: 500ms silence, 100ms padding, etc.

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset
            if "start" in res and "end" not in res:
                self.status = "voice"
                send_audio = self.audio_buffer[frame:]
                self.online.init(
                    offset=(frame + self.buffer_offset) / self.SAMPLING_RATE
                )
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
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM.
                self.buffer_offset += max(
                    0, len(self.audio_buffer) - self.SAMPLING_RATE
                )
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE :]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif (
            self.current_online_chunk_buffer_size
            > self.SAMPLING_RATE * self.online_chunk_size
        ):
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            logger.debug("no online update, only VAD")
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret
