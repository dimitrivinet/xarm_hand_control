import json
import os
import time
from collections import deque
from statistics import mean, mode
from typing import Any, Callable, Iterable, Tuple, Union

import cv2
import dotenv
import joblib
import mediapipe as mp
import numpy as np
import torch
import onnxruntime

from training.model import HandsClassifier
from utils import Command
from utils import ClassificationMode as Mode

dotenv.load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR')
MLP_MODEL_PATH = os.getenv('MLP_MODEL_PATH')
ONNX_MODEL_PATH = os.getenv('ONNX_MODEL_PATH')
RF_MODEL_PATH = os.getenv('RF_MODEL_PATH')

# choose classification mode
# with env variables:
"""
MODE = os.getenv('MODE')
MODE = Mode.get(MODE)
"""

# with hard-coded value:
MODE = Mode.NO_CLASSIFICATION
# MODE = Mode.RANDOM_FOREST
# MODE = Mode.MLP
# MODE = Mode.ONNX

VIDEO_INDEX = 6
WINDOW_NAME = "win"

ROBOT_COMMAND_SCALE = 100
ROBOT_SPEED = 100.0
ROBOT_MVACC = 1000.0

MAX_NUM_HANDS = 1

classification_buffer = deque(maxlen=5)
fps_buffer = deque(maxlen=50)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def class_as_str(classes: dict, class_index: int) -> str:
    """Get class name from class index
    """

    return classes[class_index]['name']


def format_landmarks(landmarks):
    """Format landmarks to the format used by selected model
    """

    ret = []

    for landmark in landmarks:
        f_landmarks = [[point.x, point.y] for point in landmark.landmark]

        if MODE == Mode.RANDOM_FOREST:
            ret.append(np.array([f_landmarks, ]))

        elif MODE == Mode.MLP:
            ret.append(torch.tensor([f_landmarks, ]))

        elif MODE == Mode.ONNX:
            ret.append(np.array([f_landmarks, ], dtype=np.float32))

    return ret


def get_onnx_model() -> Callable[[np.ndarray], list]:
    """Create the onnx session and return callable used to run inference

    Returns:
        Callable[[np.ndarray], list]: function to run inference with.
        Parameter is np.ndarray with shape (1, x, y) and dtype np.float32.
    """

    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def run(data):
        result = session.run([output_name], {input_name: data})

        return result

    return run


def load_model(classes: dict = None) -> Any:
    """Load model according to selected inference mode.

    Args:
        classes (dict, optional): Classes dict. Defaults to None.

    Returns:
        Any: Model according to selection
    """

    model = None

    if MODE == Mode.RANDOM_FOREST:
        model = joblib.load(RF_MODEL_PATH)

    elif MODE == Mode.MLP:
        n_classes = len(classes)
        model = HandsClassifier(n_classes)
        model.load_state_dict(torch.load(MLP_MODEL_PATH))
        model.eval()

    elif MODE == Mode.ONNX:
        model = get_onnx_model()

    return model


def run_inference(classes: dict, landmarks: Any, model: Any) -> Union[None, list]:
    """Run inference on array of landmarks with selected model

    Args:
        classes (dict): Classes dict
        landmarks (Any): landmarks array
        model (Any): model selected with MODE

    Returns:
        list: list of string representing detected classes, buffered to
        avoid artefacts, None if MODE = Mode.NO_CLASSIFICATION
    """

    if MODE == Mode.NO_CLASSIFICATION:
        return None

    classified_hands = []
    f_landmarks = format_landmarks(landmarks)

    for landmark in f_landmarks:

        if MODE == Mode.RANDOM_FOREST:
            class_index = model.predict(landmark.reshape(1, -1))[0]

        elif MODE == Mode.MLP:
            class_index = torch.argmax(model(landmark)).item()

        elif MODE == Mode.ONNX:
            result = model(landmark)
            class_index = np.argmax(result[0].squeeze(axis=0))

        classified_hands.append(class_as_str(classes, class_index))

    # add to buffer and return most common occurence in last n frames
    classification_buffer.appendleft(tuple(classified_hands))

    return list(mode(classification_buffer))


def run_hands(image: Any, hands: mp_hands.Hands) -> Tuple[Any, list]:
    """Run hand landmark recognition on image

    Args:
        image (Any): Image to run recognition on
        hands (mp_hands.Hands): Mediapipe Hands instance

    Returns:
        annotated_image (Any): Image annotated with hand landmarks
        results.multi_hand_landmarks (list): hand landmarks as list
    """

    # Convert the BGR image to RGB, flip the image around y-axis for correct
    # handedness output and process it with MediaPipe Hands.
    results = hands.process(
        cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = cv2.flip(image.copy(), 1)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmark_style(),
            drawing_styles.get_default_hand_connection_style())

    return annotated_image, results.multi_hand_landmarks


def get_polar_coords(im_shape: Tuple, landmarks: list) -> Tuple[float, float]:
    """Translate landmarks to polar coordinates with center of palm as middle point

    Args:
        im_shape (Tuple): Shape of the image used by Mediapipe Hands
        landmarks (list): Hand landmarks

    Returns:
        dist (float): Distance from center of palm to center of image
        angle (float): Angle from vertical axis to center of palm
    """

    im_height = im_shape[0]
    im_width = im_shape[1]
    im_center = (im_width // 2, im_height // 2)

    # palm center as the point between wrist and index metacarpal head
    palm_centers = []
    for landmark in landmarks:
        p1 = (landmark.landmark[mp_hands.HandLandmark.WRIST].x * im_width,
              landmark.landmark[mp_hands.HandLandmark.WRIST].y * im_height)
        p2 = (landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * im_width,
              landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * im_height)

        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        palm_centers.append(center)

    dists_and_angles = []
    for center in palm_centers:
        x = center[0] - im_center[0]
        # normal y is from top to bottom, multiply by -1 to flip
        y = (center[1] - im_center[1]) * -1

        dist = np.linalg.norm(np.array([x, y]))

        # angle = 90 or 270 deg for x = 0
        angle = np.arctan2(y, x)

        # map angle from -pi, pi to 0, 360 with 0 on the right
        if angle < 0:
            angle += 2*np.pi
        # angle = np.interp(angle, (0, 2 * np.pi), (0, 360))
        angle = np.rad2deg(angle)

        dists_and_angles.append([dist, angle])

    dists_and_angles = np.array(dists_and_angles)
    # get index of row with smallest distance to center (ignore angle)
    min_index = np.argmin(dists_and_angles, axis=0)[0]
    dist, angle = dists_and_angles[min_index]

    return dist, angle


def get_robot_command(im_shape: Tuple, dist: float, angle: float) -> Command:
    """Translate distance and angle to an xArm command

    Args:
        im_shape (Tuple): Shape of the image used by Mediapipe Hands
        dist (float): Distance from center of palm to center of image
        angle (float): Angle from vertical axis to center of palm

    Returns:
        Command: Command NamedTuple containing fields for xArm move command
    """

    command = Command()

    # if center of palm is inside center circle
    if dist < (np.min(im_shape[:2]) / 15):
        empty_command = Command()
        return empty_command

    im_h, im_w = im_shape[0], im_shape[1]
    scaled_x = \
        dist * np.cos(np.deg2rad(angle)) / (im_w / 2) * ROBOT_COMMAND_SCALE
    scaled_y = \
        dist * np.sin(np.deg2rad(angle)) / (im_h / 2) * ROBOT_COMMAND_SCALE

    command = Command(
        x=int(scaled_x),
        y=int(scaled_y),
        speed=ROBOT_SPEED,
        mvacc=ROBOT_MVACC
    )

    return command


def run_processing(classes: dict, model: Any, image: Any, landmarks: list
                   ) -> Tuple[str, Command]:
    """Processing loop after Mediapipe Hands ran

    Args:
        classes (dict): Classes dict
        model (Any): Model according to selection
        to_show (Any): Image Mediapipe Hands ran on
        landmarks (list): Hand landmarks

    Returns:
        to_show_text (str): Text containing hand classes and distance and angle
        robot_command (Command): Command NamedTuple for xArm movement
    """

    if landmarks is None:
        return "", None

    classified_hands = run_inference(classes, landmarks, model)

    dist, angle = get_polar_coords(image.shape, landmarks)


    if classified_hands is None:
        to_show_text = f'{dist:.2f}, {angle:.2f}'
    else:
        classified_hands = ', '.join(classified_hands)
        to_show_text = " | ".join(
            [classified_hands, f'{dist:.2f}, {angle:.2f}', ])

    robot_command = get_robot_command(image.shape, dist, angle)

    return to_show_text, robot_command


def get_classes(dataset_path: os.PathLike = None) -> dict:
    """Get classes from dataset JSON

    Args:
        dataset_path (os.PathLike, optional): Path to dataset JSON. Defaults to None.

    Returns:
        dict: Classes dict
    """

    if dataset_path is None:
        dataset_path = os.path.join(DATASET_DIR, "dataset.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    classes = dataset['classes']

    return classes


def get_fps(prev_frame_time: float) -> Tuple[float, str]:
    """Calculate FPS

    Args:
        prev_frame_time (float): Time when measured at previous loop

    Returns:
        new_frame_time (float): Time when measured for current loop
        mean_fps_str (str): FPS as String, buffered to avoid changing too often
    """

    new_frame_time = time.perf_counter()

    # calculate fps with mean of last 10 fps
    fps = 1/(new_frame_time-prev_frame_time)
    fps_buffer.appendleft(int(fps))
    mean_fps_str = str(int(mean(fps_buffer)))

    return new_frame_time, mean_fps_str


def main():
    """Main loop. Captures video from camera, runs Mediapipe Hands and runs
    processing before showing image

    Raises:
        IOError: if OpenCV can't access camera
    """

    prev_frame_time = 0.0
    win = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    classes = get_classes()
    model = load_model(classes)

    cap = cv2.VideoCapture(VIDEO_INDEX)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        # Run MediaPipe Hands.
        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=0.7) as hands:

            while cap.isOpened():
                cap_ok, frame = cap.read()
                if not cap_ok:
                    continue

                ret_frame, landmarks = run_hands(frame, hands)

                to_show = cv2.flip(
                    frame, 1) if ret_frame is None else ret_frame

                to_show_text, robot_command = run_processing(
                    classes, model, to_show, landmarks)

                # show dot at center of image
                im_center = (
                    int(to_show.shape[1] / 2), int(to_show.shape[0] / 2))
                to_show = cv2.circle(to_show,
                                     im_center,
                                     radius=3,
                                     color=(0, 0, 255),
                                     thickness=3)

                prev_frame_time, mean_fps = get_fps(prev_frame_time)

                # add info to screen
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (20, 450)
                fontScale = 0.8
                fontColor = (255, 255, 255)
                tickness = 2
                cv2.putText(img=to_show,
                            text=to_show_text,
                            org=bottomLeftCornerOfText,
                            fontFace=font,
                            fontScale=fontScale,
                            color=fontColor,
                            thickness=tickness)

                cv2.putText(img=to_show,
                            text=mean_fps,
                            org=(7, 30),
                            fontFace=font,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA)

                cv2.imshow(WINDOW_NAME, to_show)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
