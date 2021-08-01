import json
import joblib
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier


def main():
    with open("./dataset.json") as f:
        dataset = json.load(f)

    data = dataset['data']
    classes = dataset['classes']

    data_ndarr = np.array(data, dtype=list)

    X, y = data_ndarr[:, 0], data_ndarr[:, 1]

    for i in range(len(X)):
        X[i] = np.array(X[i]).reshape(-1)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(list(X), list(y))

    # test model
    pred = clf.predict([[0]*42])
    print(classes[pred[0]]['name'])

    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = os.path.join(output_dir, 'random_forest.joblib')
    joblib.dump(clf, output_file_path, compress=3)

if __name__ == "__main__":
    main()
