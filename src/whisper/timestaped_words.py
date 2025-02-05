from typing import List

import logging
logger = logging.getLogger(__name__)

class TimeStampedSegment:
    """
    Represents a segment of text with start and end timestamps.

    Attributes:
        start (float): The start time of the segment.
        end (float): The end time of the segment.
        text (str): The text of the segment.
    """
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return f'{self.start} - {self.end}: {self.text}'
    
    def __repr__(self):
        return self.__str__()
    
    def shift(self, shift: float) -> None:
        """
        Shifts the segment by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segment.

        Returns:
            TimeStampedSegment: A new segment shifted by the given amount of time.

        Example:
            >>> segment = TimeStampedSegment(0.0, 1.0, "Hello")
            >>> segment.shift(1.0)
            1.0 - 2.0: Hello
        """
        self.start += shift
        self.end += shift
    
    def append_text(self, text: str):
        """
        Appends text to the segment.

        Args:
            text (str): The text to append.

        Example:
            >>> segment = TimeStampedSegment(0.0, 1.0, "Hello")
            >>> segment.append_text("!")
            >>> segment
            0.0 - 1.0: Hello!
        """
        self.text += text
    
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.text == other.text
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.shift(other)
        elif isinstance(other, str):
            return TimeStampedSegment(self.start, self.end, self.text + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

class TimeStampedSequence(list):
    """
    Represents a collection of TimeStampedSegment instances.
    """
    def __init__(self, segments: List[TimeStampedSegment] = None,sep=None):
        segments = segments if segments is not None else []
        super().__init__(segments)
        
        if sep is None:
            self.sep = self.infer_sep()


    def infer_sep(self) -> None:
        if len(self) <= 3:
            
            self.sep = None
        else:
            n_tailing_spaces = 0
            for word in self:
                if word.text.endswith(" "):
                    n_tailing_spaces += 1


            if n_tailing_spaces / len(self) > 0.5:
                sep=""
            else:
                sep= " "

            logger.debug(f"{n_tailing_spaces} words of {len(self)} have tailing spaces I will use '{sep}' to join them")
            
            self.sep = sep

    def concatenate(self) -> TimeStampedSegment:
        """
        Concatenates all segments in the sequence into a single segment.

        Returns:
            TimeStampedSegment: A single segment containing all text from the sequence.

        Example:
            >>> sequence = TimeStampedSequence([TimeStampedSegment(0.0, 1.0, "Hello"), TimeStampedSegment(1.0, 2.0, "World")])
            >>> sequence.concatenate()
            0.0 - 2.0: HelloWorld
        """
        if len(self) == 0:
            return TimeStampedSegment()
        
        sep = " " if self.sep is None else self.sep

        return TimeStampedSegment(self[0].start, self[-1].end, sep.join(segment.text for segment in self))
    
    def get_text(self,sep=None) -> str:
        """
        Returns the text of all segments in the sequence.

        Returns:
            str: The text of all segments in the sequence.

        Example:
            >>> sequence = TimeStampedSequence([TimeStampedSegment(0.0, 1.0, "Hello"), TimeStampedSegment(1.0, 2.0, "World")])
            >>> sequence.get_text(sep=' ')
            'Hello World'
        """
        if sep is None:
            if self.sep is None:
                sep = " "
            else:
                sep = self.sep

        
        return sep.join(segment.text for segment in self)
    
    def shift(self, shift: float) -> None:
        """
        Shifts all segments in the sequence by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segments.

        Returns:
            TimeStampedSequence: A new sequence with all segments shifted by the given amount of time.

        Example:
            >>> sequence = TimeStampedSequence([TimeStampedSegment(0.0, 1.0, "Hello"), TimeStampedSegment(1.0, 2.0, "World")])
            >>> sequence.shift(1.0)
            [1.0 - 2.0: Hello, 2.0 - 3.0: World]
        """
        for segment in self:
            segment.shift(shift)
    
    def __repr__(self):

        if len(self) == 0:
            return "TimeStampedSequence([])"

        representation = f"TimeStampedSequence({n} words from {start} to {end}: {first}…{last})".format(
            n= len(self),
            start= self[0].start,
            end= self[-1].end,
            first= self[0].text,
            last= self[-1].text
        )

        return representation
    
    def __str__(self):
        
        if len(self) == 0:
            return "TimeStampedSequence([])"

        return self.concatenate().__str__()






if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)