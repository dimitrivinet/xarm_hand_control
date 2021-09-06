import json
import os
from collections import deque
from statistics import mode
from typing import Any, Callable, Tuple, Union

import cv2
import joblib
import mediapipe as mp
import numpy as np

from xarm_hand_control.modules.utils import FPS
from xarm_hand_control.modules.utils import ClassificationMode
from xarm_hand_control.modules.utils import ClassificationMode as Mode
from xarm_hand_control.modules.utils import Command

# * ----------------------------------------------------------------------------
# * PROGRAM PARAMETERS
# * ----------------------------------------------------------------------------
ROBOT_COMMAND_SCALE = 100
ROBOT_SPEED = 100.0
ROBOT_MVACC = 1000.0

MAX_NUM_HANDS = 1
# * ----------------------------------------------------------------------------

WINDOW_NAME = "Hand Control"

classification_buffer = deque(maxlen=5)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def class_as_str(classes: dict, class_index: int) -> str:
    """Get class name from class index
    """

    return classes[class_index]['name']


def format_landmarks(classification_mode: ClassificationMode, landmarks: Any):
    """Format landmarks to the format used by selected model
    """

    ret = []

    for landmark in landmarks:
        f_landmarks = [[point.x, point.y] for point in landmark.landmark]

        if classification_mode == Mode.RANDOM_FOREST:
            ret.append(np.array([f_landmarks, ]))

        elif classification_mode == Mode.MLP:
            import torch

            ret.append(torch.tensor([f_landmarks, ]))

        elif classification_mode == Mode.ONNX:
            ret.append(np.array([f_landmarks, ], dtype=np.float32))

    return ret


def get_onnx_model(onnx_model_path: os.PathLike) -> Callable[[np.ndarray], list]:
    """Create the onnx session and return callable used to run inference

    Returns:
        Callable[[np.ndarray], list]: function to run inference with.
        Parameter is np.ndarray with shape (1, x, y) and dtype np.float32.
    """

    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def run(data):
        result = session.run([output_name], {input_name: data})

        return result

    return run


def load_model(classification_mode: ClassificationMode, model_path: os.PathLike, classes: dict = None) -> Any:
    """Load model according to selected inference mode.

    Args:
        classes (dict, optional): Classes dict. Defaults to None.

    Returns:
        Any: Model according to selection
    """

    model = None

    if classification_mode == Mode.RANDOM_FOREST:
        model = joblib.load(model_path)

    elif classification_mode == Mode.MLP:
        import torch
        from xarm_hand_control.modules.training.model import HandsClassifier

        n_classes = len(classes)
        model = HandsClassifier(n_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif classification_mode == Mode.ONNX:
        model = get_onnx_model(model_path)

    return model


def run_inference(classification_mode: ClassificationMode, classes: dict, landmarks: Any, model: Any) -> Union[None, list]:
    """Run inference on array of landmarks with selected model

    Args:
        classes (dict): Classes dict
        landmarks (Any): landmarks array
        model (Any): model selected with MODE

    Returns:
        list: list of string representing detected classes, buffered to
        avoid artefacts, None if MODE = Mode.NO_CLASSIFICATION
    """

    if classification_mode == Mode.NO_CLASSIFICATION:
        return None

    classified_hands = []
    f_landmarks = format_landmarks(classification_mode, landmarks)

    for landmark in f_landmarks:

        if classification_mode == Mode.RANDOM_FOREST:
            class_index = model.predict(landmark.reshape(1, -1))[0]

        elif classification_mode == Mode.MLP:
            import torch

            class_index = torch.argmax(model(landmark)).item()

        elif classification_mode == Mode.ONNX:
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


def get_center_coords(landmarks: list) -> Tuple[float, float]:
    """Translate landmarks to cartesian coordinates with center of palm as middle point

    Args:
        landmarks (list): Hand landmarks

    Returns:
        x (float): Center of palm x coordinate from center of image
        y (float): Center of palm y coordinate from center of image
    """

    # palm center as the point between wrist and index metacarpal head
    palm_centers = []
    for landmark in landmarks:
        p1 = (landmark.landmark[mp_hands.HandLandmark.WRIST].x,
              landmark.landmark[mp_hands.HandLandmark.WRIST].y)
        p2 = (landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
              landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)

        palm_center = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        palm_center_centered = [palm_center[0] - 0.5, - (palm_center[1] - 0.5)]
        palm_centers.append(palm_center_centered)

    palm_centers_distances = [np.linalg.norm(
        palm_center, ord=2) for palm_center in palm_centers]
    # get index of row with smallest distance to center (ignore angle)
    min_index = np.argmin(palm_centers_distances, axis=0)
    x, y = palm_centers[min_index]

    return x, y


def get_robot_command(x: float, y: float) -> Command:
    """Translate x and y to an xArm command

    Args:
        x (float): Center of palm x coordinate from center of image
        y (float): Center of palm y coordinate from center of image

    Returns:
        Command: Command NamedTuple containing fields for xArm move command
    """

    command = Command()

    dist = np.linalg.norm([x, y], ord=2)

    # if center of palm is inside center circle
    if dist < 0.1:
        empty_command = Command()
        return empty_command

    scaled_x = x * ROBOT_COMMAND_SCALE
    scaled_y = y * ROBOT_COMMAND_SCALE

    command = Command(
        x=scaled_x,
        y=scaled_y,
        speed=ROBOT_SPEED,
        mvacc=ROBOT_MVACC
    )

    return command


def run_processing(classification_mode: ClassificationMode, classes: dict, model: Any, image: Any, landmarks: list
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

    classified_hands = run_inference(
        classification_mode, classes, landmarks, model)

    x, y = get_center_coords(landmarks)

    if classified_hands is None:
        to_show_text = f'{x:.2f}, {y:.2f}'
    else:
        classified_hands = ', '.join(classified_hands)
        to_show_text = " | ".join(
            [classified_hands, f'{x:.2f}, {y:.2f}', ])

    robot_command = get_robot_command(x, y)

    return to_show_text, robot_command


def get_classes(dataset_path: os.PathLike) -> dict:
    """Get classes from dataset JSON

    Args:
        dataset_path (os.PathLike, optional): Path to dataset JSON.
    Returns:
        dict: Classes dict
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    classes = dataset['classes']

    return classes


def add_image_info(image, top_left_text, bottom_left_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_of_text = (20, 30)
    bottom_left_corner_of_text = (20, image.shape[0] - 30)
    font_scale = 0.8
    white = (255, 255, 255)
    red = (0, 0, 255)
    tickness = 2
    linetype = cv2.LINE_AA

    # show fps
    cv2.putText(img=image,
                text=top_left_text,
                org=top_left_corner_of_text,
                fontFace=font,
                fontScale=font_scale,
                color=red,
                thickness=tickness,
                lineType=linetype)

    # show hand info
    cv2.putText(img=image,
                text=bottom_left_text,
                org=bottom_left_corner_of_text,
                fontFace=font,
                fontScale=font_scale,
                color=white,
                thickness=tickness,
                lineType=linetype)

    # show dot at center of image
    im_center = (
        int(image.shape[1] / 2), int(image.shape[0] / 2))
    cv2.circle(img=image,
               center=im_center,
               radius=3,
               color=(0, 0, 255),
               thickness=3)


def process(classification_mode: ClassificationMode = ClassificationMode.NO_CLASSIFICATION,
            video_index: int = 0,
            dataset_path: os.PathLike = None,
            model_path: os.PathLike = None
            ):
    """Main loop. Captures video from camera, runs Mediapipe Hands and runs
    processing before showing image

    Raises:
        IOError: if OpenCV can't access camera
    """
    # global MODE
    # MODE = classification_mode

    # global VIDEO_INDEX
    # VIDEO_INDEX = video_index

    inner_fps = FPS()
    outer_fps = FPS()

    if dataset_path is not None:
        classes = get_classes(dataset_path)
    else:
        classes = None

    model = load_model(classification_mode, model_path, classes)

    win = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(video_index)
    W, H = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=0.7)

        while cap.isOpened():
            cap_ok, frame = cap.read()
            if not cap_ok:
                print("cap not ok")
                continue

            inner_fps.update()

            ret_frame, landmarks = run_hands(frame, hands)

            to_show = cv2.flip(
                frame, 1) if ret_frame is None else ret_frame

            to_show_text, robot_command = run_processing(
                classification_mode, classes, model, to_show, landmarks)

            inner_fps.update()
            outer_fps.update()
            outer_fps_value = int(outer_fps.fps())
            inner_fps_value = int(inner_fps.fps())

            fpss = f'{outer_fps_value}/{inner_fps_value}'

            add_image_info(to_show, fpss, to_show_text)

            cv2.imshow(WINDOW_NAME, to_show)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
