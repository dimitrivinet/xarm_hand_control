# xarm_hand_control

Hand landmark recognition with Mediapipe and landmark classification with PyTorch

## Setup

Optional (requires python3-venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then:

```bash
pip3 install git+https://github.com/dimitrivinet/xarm_hand_control
```

And test with:

```bash
python3 -m xarm_hand_control
```

## Dependencies (optional)

- [PyTorch](https://pytorch.org/get-started/locally/)

- onnxruntime (`pip3 install onnxruntime`)

## Download trained models

Either clone the repository and run `make get_models` or:

```bash
wget https://github.com/dimitrivinet/xarm_hand_control/releases/download/v1.0/models.zip
unzip models.zip -d models
```

And cleanup:

```bash
rm models.zip
rm models.zip
```
