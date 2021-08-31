import os
import json
import random

from modules.training.acquire import classes


def split_dataset(dataset_path: os.PathLike, train_percentage: float):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    data = dataset['data']
    random.shuffle(data)

    n_training_data = int(len(data) * train_percentage)
    trainset = {'data': data[:n_training_data], 'classes': classes}
    validset = {'data': data[n_training_data:], 'classes': classes}

    dataset_dir = os.path.dirname(dataset_path)

    with open(os.path.join(dataset_dir, 'dataset_train.json'), 'w') as f:
        json.dump(trainset, f)

    with open(os.path.join(dataset_dir, 'dataset_valid.json'), 'w') as f:
        json.dump(validset, f)

    print("done.")
