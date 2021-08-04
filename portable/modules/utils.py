from enum import Enum, auto
from collections import namedtuple


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

    @staticmethod
    def get(label):
        ret = None

        try:
            ret = ClassificationMode[label]
        except KeyError:
            ret = ClassificationMode.NONE

        return ret

_command_fields = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'speed', 'mvacc']
Command = namedtuple(typename='Command',
                     field_names=_command_fields,
                     defaults=[None]*8)
