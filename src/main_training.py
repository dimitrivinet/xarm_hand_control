import os

import dotenv

from training.acquire import acquire
from training.export import export
from training.split_dataset import split_dataset
from training.train import train
from utils import TrainingMode as Mode

dotenv.load_dotenv()

OUTPUT_DIR = os.getenv('OUTPUT_DIR')
MLP_MODEL_PATH = os.getenv('MLP_MODEL_PATH')
DATASET_DIR = os.getenv('DATASET_DIR')


# uncomment desired mode
# MODE = Mode.TRAIN
MODE = Mode.EXPORT
# MODE = Mode.ACQUIRE

SAVE_ALL = False  # True: save all checkpoints

VIDEO_INDEX = 6

NUM_EPOCHS = 1000


def main():
    if MODE == Mode.TRAIN:

        if SAVE_ALL:
            checkpoints_dir_path = os.path.join(OUTPUT_DIR, "checkpoints")
            if not os.path.exists(checkpoints_dir_path):
                print(f"crating dir {checkpoints_dir_path}")
                os.makedirs(checkpoints_dir_path, exist_ok=True)

        train(OUTPUT_DIR, NUM_EPOCHS, SAVE_ALL)

    elif MODE == Mode.EXPORT:
        export()

    elif MODE == Mode.ACQUIRE:
        DATASET_PATH = os.path.join(DATASET_DIR, "dataset.json")
        acquire(DATASET_PATH, VIDEO_INDEX)
        split_dataset(DATASET_PATH, train_percentage=0.7)


    else:
        print("mode not recognized")


if __name__ == "__main__":
    main()
