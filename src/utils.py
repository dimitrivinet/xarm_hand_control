from enum import Enum, auto
from collections import namedtuple


class TrainingMode(Enum):
    """Program mode enum
    """
    TRAIN = auto()
    EXPORT = auto()
    ACQUIRE = auto()

class ClassificationMode(Enum):
    NO_CLASSIFICATION = auto()
    RANDOM_FOREST = auto()
    MLP = auto()

    @staticmethod
    def get(label, default="NO_CLASSIFICATION"):
        try:
            ret = Mode[label]
        except KeyError:
            try:
                ret = Mode[default]
            except KeyError:
                ret = Mode.NO_CLASSIFICATION
        finally:
            return ret

_command_fields = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'speed', 'mvacc']
Command = namedtuple(typename='Command',
                     field_names=_command_fields,
                     defaults=[None]*8)
