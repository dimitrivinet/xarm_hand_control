import os
import json
import random

random.seed(42)

DIRNAME = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(DIRNAME, "dataset/dataset.json")


def split_dataset(train_percentage: float):
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    data = dataset['data']
    random.shuffle(data)

    n_training_data = int(len(data) * train_percentage)
    trainset = {'data': data[:n_training_data]}
    validset = {'data': data[n_training_data:]}

    dataset_dirname = os.path.dirname(DATASET_PATH)

    with open(os.path.join(dataset_dirname, "dataset_train.json"), 'w') as f:
        json.dump(trainset, f)

    with open(os.path.join(dataset_dirname, "dataset_valid.json"), 'w') as f:
        json.dump(validset, f)

    print("done.")


def main():
    split_dataset(0.7)

if __name__ == "__main__":
    main()
