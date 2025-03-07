from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple

import logging
logger = logging.getLogger(__name__)

@dataclass
class TimedText:
    start: Optional[float]
    end: Optional[float]
    text: Optional[str] = ''
    speaker: Optional[int] = -1
    probability: Optional[float] = None

    def shift(self, offset: float) -> None:
        self.start += offset
        self.end += offset

@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, self.probability)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    pass

@dataclass
class SpeakerSegment(TimedText):
    pass


class TimedList(list):
    """
    Represents a collection of TimedText instances.
    """
    def __init__(self, words: List[TimedText] = None,sep=None):
        words = words if words is not None else []
        super().__init__(words)
        
        self.sep= sep




    def infer_sep(self) -> None:
        if len(self) == 0:
            return " "

        n_tailing_spaces = 0
        for word in self:
            if word.text.endswith(' ') or word.text.startswith(' '):
                n_tailing_spaces += 1


        if n_tailing_spaces / len(self) > 0.8:
            sep=""
        else:
            sep= " "
        
        logger.debug(f"{n_tailing_spaces} words of {len(self)} have tailing spaces I will use '{sep}' to join them")
        self.sep = sep

    def concatenate(self, sep: Optional[str] = None, offset: float = 0) -> TimedText:
        """
        Concatenates all segments in the sequence into a single segment.

        Returns:
            TimedText: A single segment containing all text from the sequence.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.concatenate() == TimedText(start=0.0, end=2.0, text='Hello World')
            True
            >>> sequence.sep==" "
            True
        """

        if len(self) == 0:
            return TimedText(None, None)
        

        text = self.get_text(sep)
        probability = sum(word.probability for word in self if word.probability) / len(self)
        if probability==0:
            probability=None

        speaker= self[0].speaker
        if not all(word.speaker==speaker for word in self):
            logger.warning(f"Not all words in the sequence are not from the same speaker: {self}")
            speaker=-1

        start = offset + self[0].start
        end = offset + self[-1].end

        return TimedText(start, end, text, probability=probability, speaker=speaker)

    
    def get_text(self,sep: Optional[str] = None) -> str:
        """
        Returns the text of all segments in the sequence.

        Returns:
            str: The text of all segments in the sequence.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.get_text()
            'Hello World'
        """

        if len(self) == 0:
            return ""

        if sep is None:
            if self.sep is None:
                self.infer_sep()
            sep = self.sep
        elif self.sep is None:
            self.sep = sep
        
        return sep.join(word.text for word in self)
    
    def shift(self, shift: float) -> None:
        """
        Shifts all segments in the sequence by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segments.

        Returns:
            TimedList: A new sequence with all segments shifted by the given amount of time.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.shift(1.0)
            >>> sequence == [TimedText(start=1.0, end=2.0, text='Hello'), TimedText(start=2.0, end=3.0, text='World')]
            True
        """
        for segment in self:
            segment.shift(shift)

    def extend(self, other: List[TimedText] | 'TimedList'):
        """
        Extend the sequence with another sequence of TimedText instances.
        """
        assert isinstance(other, TimedList), "Expect a TimedList to extend"
        
        if len(other) == 0:
            return
        elif len(self) == 0:
            super().extend(other)
            self.sep = other.sep
            return
        else:
            if self.sep != other.sep:
                logger.warning(f"Two TimedLists with different separators are being extended: {self.sep} != {other.sep} use ' ' in future")
                self.sep = ' '
            super().extend(other)
        
        

    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):

        if len(self) == 0:
            return "[]"
        if len(self) <= 5:
            return f"[{self.get_text()} ({self[0].start:.3f} - {self[-1].end:.3f})]"

        return "[{first}…{last} ({n} words {start:.3f} - {end:.3f})]".format(
            n= len(self),
            start= self[0].start,
            end= self[-1].end,
            first= self.sep.join((self[0],self[1])),
            last= self.sep.join((self[-2],self[-1]))
        )

    def __getitem__(self, index: int) -> TimedText:
        if isinstance(index, slice):
            return TimedList(super().__getitem__(index), sep=self.sep)
        else:
            return super().__getitem__(index)

    def split_at(self, time: float, edge_word_goes_left: bool) -> Tuple['TimedList', 'TimedList']:
        """
        Split the sequence at a given time. 
        Assuming time is in the middle of a word, the flag edge_word_goes_left decides which side the word goes to.
        
        Returns a tuple of two TimedLists.

        Example:
            >>> greeting = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "all"), TimedText(2.0, 3.0, "Together!")])
            >>> before,after= greeting.split_at(1.5, edge_word_goes_left=True)
            >>> before.get_text()
            'Hello all'
            >>> before,after= greeting.split_at(1.5, edge_word_goes_left=False)
            >>> before.get_text()
            'Hello'
        """
        # word at i goes to the right list
        if edge_word_goes_left:
            # find the first word that starts after the time
            for i, word in enumerate(self):
                if word.start >= time:
                    break
            else:
                return self, TimedList([])

           
            
            
        else: # edge_word_goes_right
            # find the first word that ends after the time
            # word at i goes to the right list
            for i, word in enumerate(self):
                if word.end >= time:
                    break
            else:
                return self, TimedList([])

        return self[:i], self[i:]

        



        


    def split_to_sentences(self, sentence_splitter: Optional[Callable[[str], List[str]] | Callable[[List[str]], List[str]]] = None) -> 'TimedList':
            """
            Converts a list of tokens to a list of Sentence objects using the provided
            sentence tokenizer.
            """
            if len(self) == 0:
                return TimedList([])
            

            if sentence_splitter is None:
                logger.warning("No sentence splitter provided, concatenating tokens")
                return self.concatenate()

            full_text = self.get_text()

            try:
                sentence_texts = self.tokenize([full_text])
            except Exception as e:
                # Some tokenizers (e.g., MosesSentenceSplitter) expect a list input.
                try:
                    sentence_texts = self.tokenize(full_text)
                except Exception as e2:
                    raise ValueError("Tokenization failed") from e2

            # Match output of sentence splitter to the input tokens
            sentences: TimedList = TimedList([])
            token_index = 0

            for sent_text in sentence_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue
                sent_tokens = TimedList([])

                accumulated = ""
                # Accumulate tokens until roughly matching the length of the sentence text.
                while token_index < len(self) and len(accumulated) < len(sent_text):
                    token = self[token_index]
                    accumulated = (accumulated + self.sep + token.text).strip() if accumulated else token.text
                    sent_tokens.append(token)
                    token_index += 1
                if sent_tokens:
                    sentences.append(sent_tokens.concatenate())
            return sentences


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)