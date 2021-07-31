import json
import os
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class HandsDataset(Dataset):
    dataset_dict: dict
    classes: list
    data: list
    n_classes: int

    def __init__(self,
                 dataset_path: os.PathLike,
                 train):

        with open(dataset_path, "r") as f:
            self.dataset_dict = json.load(f)

        self.train = train
        self.classes = self.dataset_dict['classes']
        self.n_classes = len(self.classes)
        self.data = np.array(self.dataset_dict['data'], dtype=object)

    def __getitem__(self, index):
        landmarks, label = self.data[index]

        return torch.tensor(landmarks), label

    def __len__(self):
        return len(self.data)

    def class_str(self, class_id: int):
        return self.classes_dict[class_id]


@dataclass
class TrainingData():
    trainset: HandsDataset = HandsDataset(
        './dataset.json',
        train=True
    )
    validset: HandsDataset = HandsDataset(
        './dataset.json',
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
