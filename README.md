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

If you used the optional step above, you can quit out of the virtual environment with `deactivate` and remove it from your machine with `rm -rf .venv`.

## Dependencies (optional)

- [PyTorch](https://pytorch.org/get-started/locally/)

- onnxruntime (`pip3 install onnxruntime`)

## Usage

See [USAGE.md](./USAGE.md)
