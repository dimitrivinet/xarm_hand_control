import os

from xarm_hand_control.modules.utils import ClassificationMode


def process(classification_mode: ClassificationMode = ClassificationMode.NO_CLASSIFICATION,
            video_index: int = 0,
            dataset_path: os.PathLike = None,
            model_path: os.PathLike = None):
    """Run processing loop: capture video, run Mediapipe Hands, run
    classification on landmarks if wanted, and show image with drawn landmarks.

    Args:
        classification_mode (ClassificationMode, optional): Classification to
        run on Mediapipe landmarks. Defaults to ClassificationMode.NO_CLASSIFICATION.
        video_index (int, optional): index of video stream
        (ex: 0 is /dev/video0). Defaults to 0.
        dataset_path (os.PathLike, optional): Path to dataset JSON file.
        See classes.json for a minimal example. Defaults to None.
        model_path (os.PathLike, optional): Path to file containing
        model to run classification with. Defaults to None.
    """

    import xarm_hand_control.modules.processing.process as mpp

    mpp.process(classification_mode, video_index, dataset_path, model_path)


def train(dataset_dir: os.PathLike,
          output_dir: os.PathLike,
          num_epochs=100,
          save_all=False):
    """Train a new MLP model.

    Args:
        dataset_dir (os.PathLike): Directory containing dataset_train.json
        and dataset_valid.json
        output_dir (os.PathLike): Directory to write checkpoints and model to.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 100.
        save_all (bool, optional): if True: save all checkpoints during
        training. Defaults to False.
    """

    import xarm_hand_control.modules.training.train as mtt

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
    """Export model as Onnx model

    Args:
        dataset_path (os.PathLike): Path to dataset JSON file. Used to get classes.
        input_path (os.PathLike): Path to model to export.
        output_path (os.PathLike): Path to save exported model to.
    """

    import xarm_hand_control.modules.training.export as mte

    mte.export(dataset_path, input_path, output_path)


def acquire(output_dir: os.PathLike, video_index: int = 0):
    """Acquire data to create dataset. Goes class by class, and gives
    unlimited time to capture hand gestures for the given class. Make sure to
    always make the correct gesture to avoid poluting your dataset with wrong data.

    Args:
        output_dir (os.PathLike): Directory to contain acquired dataset, as
        well as train and valid splits for this dataset.
        video_index (int): index of video stream
        (ex: 0 is /dev/video0). Defaults to 0.
    """

    import xarm_hand_control.modules.training.acquire as mta

    mta.acquire(output_dir, video_index)
    input_dataset_path = os.path.join(output_dir, "dataset.json")
    mta.split_dataset(input_dataset_path, output_dir, train_percentage=0.7)
