
import numpy as np
from collections import namedtuple

class TimeStampedSegment():
    def __init__(self, start=None, end=None, text=""):
        self.start = start
        self.end = end
        self.text = text

    def __getitem__(self, key):
        if key == 0:
            return self.start
        elif key == 1:
            return self.end
        elif key == 2:
            return self.text
        elif isinstance(key, slice):
            raise NotImplementedError('Slicing not supported')
        
    def __str__(self):
        return f'{self.start} - {self.end}: {self.text}'
    
    def __repr__(self):
        return self.__str__()
    
    def shift(self, shift):
        return TimeStampedSegment(self.start + shift, self.end + shift, self.text)
    
    def append_text(self, text):
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
    
    
    

class TimeStampedText(list):

    def __init__(self, time_stamped_segments: list[TimeStampedSegment]):
        super().__init__(time_stamped_segments)
        self._index = 0

    def words(self):
        return [segment.text for segment in self]
    def starts(self):
        return [segment.start for segment in self]
    
    def ends(self):
        return [segment.end for segment in self]
    

    def concatenate(self, sep:str, offset=0)->TimeStampedSegment:
        """
        Concatenates the timestamped words or sentences into a single sequence with timing information.
        This method joins all words in the sequence using the specified separator and preserves 
        the timing information from the first to the last word.
        Args:
            sep (str): Separator string used to join the words together
            offset (float, optional): Time offset to add to begin/end timestamps. Defaults to 0.
        Returns:
            TimeStampedSegment: A new segment containing:
                - Start time: First word's start time + offset 
                - End time: Last word's end time + offset
                - Text: All words joined by separator
        Examples:
            >>> seg = TimeStampedSegment([(1.0, 2.0, "hello"), (2.1, 3.0, "world!")])
            >>> result = seg.concatenate(" ")
            >>> print(result)
            (1.0, 3.0, "hello world!")
        Notes:
            Returns an empty TimeStampedSegment if the current segment contains no words.
        """
  

        if len(self) == 0:
            return TimeStampedSegment()

        combined_text = sep.join(self.words())

        b = offset + self[0][0]
        e = offset + self[-1][1]
        return TimeStampedSegment(b, e, combined_text)



