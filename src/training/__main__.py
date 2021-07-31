
import argparse
import os

from training.acquire import acquire
from training.export import export
from training.train import train

parser = argparse.ArgumentParser(description='Train hands classifier')

parser.add_argument('-o', '--output-path',
                    type=str,
                    default="./trained_models",
                    help="path to save models to (train), path to export model to (export)",
                    )
parser.add_argument('-i', '--input-path',
                    type=str,
                    default="./trained_models/best.pt",
                    help="path to trained model",
                    )
parser.add_argument('-e', '--num-epochs',
                    type=int,
                    default=3,
                    help="number of epochs to train for (default: 3)",
                    )
parser.add_argument('-a', '--save-all',
                    action='store_true',
                    help="save all checkpoints",
                    )
parser.add_argument('-v', '--video',
                    type=int,
                    default=0,
                    help='video source index')
parser.add_argument('--mode', type=str,
                    help="module mode [acquire, train, export]")
args = parser.parse_args()
# print(args)

TRAIN = args.mode == "train"
EXPORT = args.mode == "export"
ACQUIRE = args.mode == "acquire"
OUTPUT_PATH = args.output_path
INPUT_PATH = args.input_path
NUM_EPOCHS = args.num_epochs
SAVE_ALL = args.save_all
VIDEO_INDEX = args.video


if TRAIN:
    output_path = os.path.join(OUTPUT_PATH, "training")
    if not os.path.exists(output_path):
        print(f"crating dir {output_path}")
        os.makedirs(output_path, exist_ok=True)

    train(OUTPUT_PATH, NUM_EPOCHS, SAVE_ALL)

elif EXPORT:
    output_path = os.path.join(OUTPUT_PATH, "export")
    assert os.path.exists(INPUT_PATH), "file not found at specified input path"

    if not os.path.exists(output_path):
        print(f"crating dir {output_path}")
        os.makedirs(output_path, exist_ok=True)

    export(INPUT_PATH, OUTPUT_PATH)

elif ACQUIRE:
    acquire(OUTPUT_PATH, VIDEO_INDEX)

else:
    print("mode not recognized")
