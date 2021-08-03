import json
import os
from dataclasses import dataclass

import dotenv
import torch
from torch.utils.data import DataLoader, Dataset

import training.user_transforms as user_T

dotenv.load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR')

TRANSFORMS = [
    user_T.flip_h,
    user_T.rotate(-15, 15),
]


def transform(landmarks: torch.Tensor) -> torch.Tensor:
    """Apply transforms to landmarks

    Args:
        landmarks (torch.tensor): Hands landmarks

    Returns:
        torch.tensor: transformed landmarks
    """
    transformed = torch.clone(landmarks)

    transformed = user_T.normalize_center(transformed)

    for transform in TRANSFORMS:
        transformed = transform(transformed)

    transformed = user_T.denormalize_center(transformed)

    return transformed


class HandsDataset(Dataset):
    """Dataset Class for mediapipe hands
    """
    train: bool
    dataset_dir: os.PathLike
    dataset_path: os.PathLike
    dataset_dict: dict
    data: list
    classes: list
    n_classes: int

    def __init__(self, dataset_dir: os.PathLike, train: bool):
        """Create Hands Dataset

        Args:
            dataset_dir (os.PathLike): Directory containing train or valid dataset
            train (bool): train mode if true, valid mode if false
        """

        self.train = train

        if self.train:
            self.dataset_path = os.path.join(dataset_dir, "dataset_train.json")
        else:
            self.dataset_path = os.path.join(dataset_dir, "dataset_valid.json")

        with open(self.dataset_path, "r") as f:
            self.dataset_dict = json.load(f)

        self.data = self.dataset_dict['data']

        self.classes = self.dataset_dict['classes']
        self.n_classes = len(self.classes)

    def __getitem__(self, index):
        landmarks, label = self.data[index]

        transformed = transform(torch.tensor(landmarks))

        return transformed, label

    def __len__(self):
        return len(self.data)

    def class_str(self, class_id: int):
        return self.classes_dict[class_id]


@dataclass
class TrainingData():
    """Class for Training Data. Contains training and validation datasets as
        well as training and validation dataloaders
    """
    trainset: HandsDataset = HandsDataset(
        DATASET_DIR,
        train=True
    )
    validset: HandsDataset = HandsDataset(
        DATASET_DIR,
        train=False
    )
    trainloader: DataLoader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    validloader: DataLoader = DataLoader(
        validset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
