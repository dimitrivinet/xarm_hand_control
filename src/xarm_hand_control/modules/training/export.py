import os
import json

import torch

from xarm_hand_control.modules.training.model import HandsClassifier


def export(dataset_path: os.PathLike,
           input_path: os.PathLike,
           output_path: os.PathLike):

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    classes = dataset['classes']
    n_classes = len(classes)

    model = HandsClassifier(n_classes)
    model.load_state_dict(torch.load(input_path))
    model.eval()

    dummy_data = torch.randn((1, 21, 2))

    with open(output_path, 'wb') as f:
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
