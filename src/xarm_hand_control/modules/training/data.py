import json
import os
import sys
import traceback
from dataclasses import dataclass

import xarm_hand_control.modules.training.user_transforms as user_T
import torch
from torch.utils.data import DataLoader, Dataset

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

        try:
            with open(self.dataset_path, "r") as f:
                self.dataset_dict = json.load(f)
        except FileNotFoundError:
            print(traceback.format_exc())
            sys.exit("Enter correct DATASET_DIR in .env or env variables")

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
    trainset: HandsDataset
    validset: HandsDataset
    trainloader: DataLoader
    validloader: DataLoader

    def __init__(self, dataset_dir):
        self.trainset = HandsDataset(
            dataset_dir,
            train=True
        )
        self.validset = HandsDataset(
            dataset_dir,
            train=False
        )
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )
        self.validloader = DataLoader(
            self.validset,
            batch_size=32,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )
