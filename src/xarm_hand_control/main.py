import os

import xarm_hand_control.modules.training.train as mtt
import xarm_hand_control.modules.training.export as mte
import xarm_hand_control.modules.training.acquire as mta
import xarm_hand_control.modules.processing.process as mpp
from xarm_hand_control.modules.utils import ClassificationMode


def process(classification_mode: ClassificationMode = ClassificationMode.NO_CLASSIFICATION,
            video_index: int = 0,
            dataset_path: os.PathLike = None,
            model_path: os.PathLike = None):

    mpp.process(classification_mode, video_index, dataset_path, model_path)


def train(dataset_dir: os.PathLike,
          output_dir: os.PathLike,
          num_epochs=100,
          save_all=False):

    if save_all:
        checkpoints_dir_path = os.path.join(
            output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir_path):
            print(f"crating dir {checkpoints_dir_path}")
            os.makedirs(checkpoints_dir_path, exist_ok=True)

    mtt.train(dataset_dir, output_dir, num_epochs, save_all)


def export(dataset_path: os.PathLike,
           input_path: os.PathLike,
           output_path: os.PathLike):

    mte.export(dataset_path, input_path, output_path)


def acquire(output_path: os.PathLike, video_index: int):
    mta.acquire(output_path, video_index)
    mta.split_dataset(output_path, train_percentage=0.7)


# if __name__ == "__main__":
#     # main()
#     classification_mode = ClassificationMode.MLP
#     dataset_path = "/home/dimitri/Documents/Projects/xarm_hand_control/classes.json"
#     model_path = "/home/dimitri/Documents/Projects/xarm_hand_control/models/best.pt"
#     process(classification_mode, video_index=6, dataset_path=dataset_path, model_path=model_path)
