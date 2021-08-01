import enum
from collections import namedtuple

class Mode(enum.Enum):
    NO_CLASSIFICATION = 0
    RANDOM_FOREST = 1
    MLP = 2


Command = namedtuple(typename='Command',
                     field_names=['x', 'y', 'z',
                                  'roll', 'pitch', 'yaw',
                                  'speed', 'mvacc'],
                     defaults=[None]*8)
