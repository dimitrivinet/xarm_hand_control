import torch
import random

import numpy as np


def normalize_center(x: torch.tensor):
    # cast origin from top left to middle of image

    torch.add(x, -0.5, out=x)

    for i, _ in enumerate(x):
        x[i, 1] = - x[i, 1]

    return x

def denormalize_center(x: torch.tensor):
    # cast origin from top left to middle of image

    for i, _ in enumerate(x):
        x[i, 1] = - x[i, 1]

    torch.add(x, 0.5, out=x)

    return x

def flip_h(x: torch.tensor):
    # randomly flip according to y axis

    do_flip_h = random.randint(0, 1)
    if do_flip_h:
        for i, _ in enumerate(x):
            x[i, 0] = - x[i, 0]

    return x

def rotate(min_angle: float, max_angle: float):
    def inner(x: torch.tensor):
        # do_rotate = random.randint(0, 1)
        do_rotate = 1
        if do_rotate:
            rot_angle = random.randint(min_angle, max_angle)

            rot_mat = torch.zeros((2, 2))
            rot_mat[0, 0] = np.cos(np.deg2rad(rot_angle))
            rot_mat[0, 1] = - np.sin(np.deg2rad(rot_angle))
            rot_mat[1, 0] = np.sin(np.deg2rad(rot_angle))
            rot_mat[1, 1] = np.cos(np.deg2rad(rot_angle))

            ret = torch.zeros_like(x)
            for i, point in enumerate(x):
                point = point.view(2, 1)
                ret[i] = torch.mm(rot_mat, point).view(2)

            return ret

        return x

    return inner

if __name__ == "__main__":
    test_tensor = torch.rand((21, 2))

    print(test_tensor[:4])

    test_tensor = normalize_center(test_tensor)
    print(test_tensor[:4])
    test_tensor = denormalize_center(test_tensor)

    print(test_tensor[:4])
