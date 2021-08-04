import os
import json

import torch
import dotenv

from training.model import HandsClassifier

dotenv.load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR')
MLP_MODEL_PATH = os.getenv('MLP_MODEL_PATH')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

def export():
    dataset_path = os.path.join(DATASET_DIR, 'dataset.json')
    onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    classes = dataset['classes']
    n_classes = len(classes)

    model = HandsClassifier(n_classes)
    model.load_state_dict(torch.load(MLP_MODEL_PATH))
    model.eval()

    dummy_data = torch.randn((1, 21, 2))

    with open(onnx_path, 'wb') as f:
        torch.onnx.export(
            model,
            dummy_data,
            f,
            do_constant_folding=True,
            export_params=True,
            input_names=["landmarks"],
            output_names=["preds"],
            opset_version=10,
            verbose=True,
        )

    print('exported model to onnx format')
