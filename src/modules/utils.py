import time
from collections import deque, namedtuple
from enum import Enum, auto
from statistics import mean


class FPS:
    def __init__(self, buflen=25):
        self._prev_time_frame = 0.0
        self._last_time_frame = 0.0
        self.buffer = deque(maxlen=buflen)

    def start(self):
        # start the timer
        self._prev_time_frame = time.perf_counter()
        return self

    def stop(self):
        # stop the timer
        self._last_time_frame = time.perf_counter()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._prev_time_frame = self._last_time_frame
        self._last_time_frame = time.perf_counter()

    def fps(self) -> float:
        # compute the (approximate) frames per second
        fps =  1. / (self._last_time_frame - self._prev_time_frame)
        self.buffer.appendleft(fps)
        return mean(self.buffer)


class TrainingMode(Enum):
    """main_training.py mode enum
    """
    NONE = auto()
    TRAIN = auto()
    EXPORT = auto()
    ACQUIRE = auto()

    @staticmethod
    def get(label):
        ret = None

        try:
            ret = TrainingMode[label]
        except KeyError:
            ret = TrainingMode.NONE

        return ret


class ClassificationMode(Enum):
    """main_procesing.py mode enum
    """
    NO_CLASSIFICATION = auto()
    RANDOM_FOREST = auto()
    MLP = auto()
    ONNX = auto()

    @staticmethod
    def get(label):
        ret = None

        try:
            ret = ClassificationMode[label]
        except KeyError:
            ret = ClassificationMode.NO_CLASSIFICATION

        return ret


_command_fields = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'speed', 'mvacc']
Command = namedtuple(typename='Command',
                     field_names=_command_fields,
                     defaults=[None]*8)
