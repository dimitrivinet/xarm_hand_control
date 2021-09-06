# xarm_hand_control

Hand landmark recognition with Mediapipe and landmark classification with PyTorch

## Setup

- Optional:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then:

```bash
pip3 install git+https://github.com/dimitrivinet/xarm_hand_control
```

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
