import json
import joblib
import os

import dotenv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

dotenv.load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR')
RF_MODEL_PATH = os.getenv('RF_MODEL_PATH')

def main():
    with open(os.path.join(DATASET_DIR, "dataset.json")) as f:
        dataset = json.load(f)

    data = dataset['data']
    classes = dataset['classes']

    data_ndarr = np.array(data, dtype=list)

    X, y = data_ndarr[:, 0], data_ndarr[:, 1]

    X = np.array(list(X), dtype=np.float32)
    X = X.reshape(X.shape[0], -1)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(list(X), list(y))

    # test model
    pred = clf.predict([[0]*42])
    print(classes[pred[0]]['name'])

    joblib.dump(clf, RF_MODEL_PATH, compress=3)

if __name__ == "__main__":
    main()
