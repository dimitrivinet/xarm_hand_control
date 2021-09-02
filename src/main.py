import os

import dotenv

from modules.training.acquire import acquire
from modules.training.export import export
from modules.training.split_dataset import split_dataset
from modules.training.train import train
from modules.processing.process import process
from modules.utils import ProgramMode, ClassificationMode

dotenv.load_dotenv()

#* ----------------------------------------------------------------------------
#* PROGRAM PARAMETERS
#* ----------------------------------------------------------------------------
SAVE_ALL = True  # True: save all checkpoints
VIDEO_INDEX = 6
NUM_EPOCHS = 1000
#* ----------------------------------------------------------------------------


OUTPUT_DIR = os.getenv('OUTPUT_DIR', default=None)
MLP_MODEL_PATH = os.getenv('MLP_MODEL_PATH')
DATASET_DIR = os.getenv('DATASET_DIR')


# choose program mode
# with env variable:
HC_PROGRAM_MODE = os.getenv('HC_PROGRAM_MODE')
if HC_PROGRAM_MODE is not None:
    HC_PROGRAM_MODE = ProgramMode.get(HC_PROGRAM_MODE)
else:
    # with hard-coded value:
    HC_PROGRAM_MODE = ProgramMode.NONE
    # HC_PROGRAM_MODE = ProgramMode.PROCESS
    # HC_PROGRAM_MODE = ProgramMode.TRAIN
    # HC_PROGRAM_MODE = ProgramMode.EXPORT
    # HC_PROGRAM_MODE = ProgramMode.ACQUIRE


# choose classification mode
# with env variable:
HC_CLASSIFICATION_MODE = os.getenv('HC_CLASSIFICATION_MODE')
if HC_CLASSIFICATION_MODE is not None:
    HC_CLASSIFICATION_MODE = ClassificationMode.get(HC_CLASSIFICATION_MODE)
else:
    # with hard-coded value:
    HC_CLASSIFICATION_MODE = ClassificationMode.NO_CLASSIFICATION
    # HC_CLASSIFICATION_MODE = ClassificationMode.RANDOM_FOREST
    # HC_CLASSIFICATION_MODE = ClassificationMode.MLP
    # HC_CLASSIFICATION_MODE = ClassificationMode.ONNX


def main():
    if HC_PROGRAM_MODE == ProgramMode.PROCESS:
        process(HC_CLASSIFICATION_MODE)

    elif HC_PROGRAM_MODE == ProgramMode.TRAIN:
        if OUTPUT_DIR == '' or OUTPUT_DIR is None:
            print("Please specify an output directory with OUTPUT_DIR in .env or env variables")
            return

        if SAVE_ALL:
            checkpoints_dir_path = os.path.join(OUTPUT_DIR, "checkpoints")
            if not os.path.exists(checkpoints_dir_path):
                print(f"crating dir {checkpoints_dir_path}")
                os.makedirs(checkpoints_dir_path, exist_ok=True)

        train(DATASET_DIR, OUTPUT_DIR, NUM_EPOCHS, SAVE_ALL)

    elif HC_PROGRAM_MODE == ProgramMode.EXPORT:
        export()

    elif HC_PROGRAM_MODE == ProgramMode.ACQUIRE:
        DATASET_PATH = os.path.join(DATASET_DIR, "dataset.json")
        acquire(DATASET_PATH, VIDEO_INDEX)
        split_dataset(DATASET_PATH, train_percentage=0.7)

    else:
        print("mode not recognized")


if __name__ == "__main__":
    main()
