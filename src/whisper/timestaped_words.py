from typing import List

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
    
    def shift(self, shift: float):
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
        return TimeStampedSegment(self.start + shift, self.end + shift, self.text)
    
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

class TimeStampedText:
    """
    Represents a collection of TimeStampedSegment instances.

    Attributes:
        segments (List[TimeStampedSegment]): The list of segments.
    """
    def __init__(self):
        self.segments: List[TimeStampedSegment] = []

    def add_segment(self, segment: TimeStampedSegment):
        """
        Adds a segment to the collection.

        Args:
            segment (TimeStampedSegment): The segment to add.

        Example:
            >>> tst = TimeStampedText()
            >>> tst.add_segment(TimeStampedSegment(0.0, 1.0, "Hello"))
            >>> tst.add_segment(TimeStampedSegment(1.0, 2.0, "world"))
            >>> len(tst)
            2
        """
        self.segments.append(segment)

    def __repr__(self):
        return f"TimeStampedText(segments={self.segments})"

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def __len__(self):
        return len(self.segments)
    
    # TODO: a function from_whisper_res()

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)