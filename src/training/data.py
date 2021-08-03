import json
import os
import random
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import user_transforms as user_T

DIRNAME = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(DIRNAME, "./dataset/dataset.json")

TRANSFORMS = [
    user_T.flip_h,
    user_T.rotate(-15, 15),
]


def transform(landmarks: torch.tensor):
    transformed = torch.clone(landmarks)

    transformed = user_T.normalize_center(transformed)

    for transform in TRANSFORMS:
        transformed = transform(transformed)

    transformed = user_T.denormalize_center(transformed)

    return transformed


class HandsDataset(Dataset):
    dataset_dict: dict
    classes: list
    data: list
    n_classes: int

    def __init__(self,
                 dataset_dir: os.PathLike,
                 train):

        self.train = train
        if self.train:
            self.dataset_dir = os.path.join(dataset_dir, "dataset_train.json")
        else:
            self.dataset_dir = os.path.join(dataset_dir, "dataset_valid.json")

        with open(dataset_dir, "r") as f:
            self.dataset_dict = json.load(f)

        self.classes = self.dataset_dict['classes']
        self.n_classes = len(self.classes)
        self.data = np.array(self.dataset_dict['data'], dtype=object)

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


transform(torch.tensor([[0.44177377223968506, 0.5780870318412781],
                        [0.37204375863075256, 0.5441620349884033],
                        [0.3180847465991974, 0.4855952858924866],
                        [0.27232617139816284, 0.4338238835334778],
                        [0.22457250952720642, 0.4062116742134094],
                        [0.3813473582267761, 0.35708481073379517],
                        [0.3614117205142975, 0.2564327120780945],
                        [0.3505905270576477, 0.19525478780269623],
                        [0.3450077474117279, 0.1415027230978012],
                        [0.4299197793006897, 0.34747517108917236],
                        [0.4234434962272644, 0.23773276805877686],
                        [0.4176733195781708, 0.1664958894252777],
                        [0.41410693526268005, 0.10712653398513794],
                        [0.47346800565719604, 0.3619852364063263],
                        [0.4828791618347168, 0.2628273665904999],
                        [0.48737889528274536, 0.19893582165241241],
                        [0.4900473654270172, 0.14461125433444977],
                        [0.5130025744438171, 0.394090861082077],
                        [0.532823920249939, 0.31909865140914917],
                        [0.5434495210647583, 0.2694242000579834],
                        [0.550909161567688, 0.22023724019527435]]))
