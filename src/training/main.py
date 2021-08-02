from enum import Enum, auto
import os

from train import train
from export import export
from acquire import acquire

DIRNAME = os.path.dirname(os.path.abspath(__file__))

class Mode(Enum):
    TRAIN = auto()
    EXPORT = auto()
    ACQUIRE = auto()

MODE = Mode.TRAIN
# MODE = Mode.EXPORT
# MODE = Mode.ACQUIRE

OUTPUT_PATH = os.path.join(DIRNAME, "./output")
INPUT_PATH = os.path.join(DIRNAME, "./output/best.pt")
SAVE_ALL = False

VIDEO_INDEX = 6

NUM_EPOCHS = 1000


if MODE == Mode.TRAIN:
    output_path = os.path.join(OUTPUT_PATH, "training")
    if not os.path.exists(output_path):
        print(f"crating dir {output_path}")
        os.makedirs(output_path, exist_ok=True)

    train(OUTPUT_PATH, NUM_EPOCHS, SAVE_ALL)

elif MODE == Mode.EXPORT:
    output_path = os.path.join(OUTPUT_PATH, "export")
    assert os.path.exists(INPUT_PATH), "file not found at specified input path"

    if not os.path.exists(output_path):
        print(f"crating dir {output_path}")
        os.makedirs(output_path, exist_ok=True)

    export(INPUT_PATH, OUTPUT_PATH)

elif MODE == Mode.ACQUIRE:
    acquire(OUTPUT_PATH, VIDEO_INDEX)

else:
    print("mode not recognized")
