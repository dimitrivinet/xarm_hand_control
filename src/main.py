import json
import sys
import joblib
import enum

import cv2
import mediapipe as mp
import numpy as np
import torch

from training.model import HandsClassifier


class Mode(enum.Enum):
    NO_CLASSIFICATION = 0
    RANDOM_FOREST = 1
    MLP = 2

# class Direction(enum.Enum):
#     NONE = 0
#     LEFT = 1
#     UP_LEFT = 2
#     UP = 3
#     UP_RIGHT = 4
#     RIGHT = 5
#     DOWN_RIGHT = 6
#     DOWN = 7
#     DOWN_LEFT = 8


# MODE = Mode.NO_CLASSIFICATION
# MODE = Mode.RANDOM_FOREST
MODE = Mode.MLP

WINDOW_NAME = "win"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def class_as_str(classes, class_index):
    return classes[class_index]['name']


def format_landmarks(landmarks):
    if landmarks is None:
        return None

    ret = []
    for landmark in landmarks:
        f_landmarks = [[point.x, point.y] for point in landmark.landmark]

        if MODE == Mode.RANDOM_FOREST:
            ret.append(np.array([f_landmarks, ]))
        elif MODE == Mode.MLP:
            ret.append(torch.tensor([f_landmarks, ]))

    return ret


def load_model(classes=None):
    model = None

    if MODE == Mode.RANDOM_FOREST:
        model = joblib.load("./output/random_forest.joblib")

    if MODE == Mode.MLP:
        # n_classes = len(classes)
        n_classes = 27
        model = HandsClassifier(n_classes)
        model.load_state_dict(torch.load('./output/best.pt'))
        model.eval()

    return model


def run_inference(classes, landmarks, model):
    classified_hands = []
    f_landmarks = format_landmarks(landmarks)

    if f_landmarks is None:
        return classified_hands

    for landmark in f_landmarks:

        if MODE == Mode.RANDOM_FOREST:
            class_index = model.predict(landmark.reshape(1, -1))[0]
        if MODE == Mode.MLP:
            class_index = torch.argmax(model(landmark)).item()

        classified_hands.append(class_as_str(classes, class_index))

    return classified_hands


def run_hands(image, hands):
    # Convert the BGR image to RGB, flip the image around y-axis for correct
    # handedness output and process it with MediaPipe Hands.
    image.flags.writeable = False
    results = hands.process(
        cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
    image.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = cv2.flip(image.copy(), 1)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmark_style(),
            drawing_styles.get_default_hand_connection_style())

    return annotated_image, results.multi_hand_landmarks
    # return cv2.flip(annotated_image, 1), results.multi_hand_landmarks


def get_robot_movement(im_shape, landmarks):
    if landmarks is None:
        return None, None

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
        angle = np.interp(angle, (0, 2 * np.pi), (0, 360))

        dists_and_angles.append([dist, angle])

    dists_and_angles = np.array(dists_and_angles)
    # get index of row with smallest distance to center (ignore angle)
    min_index = np.argmin(dists_and_angles, axis=0)[0]
    dist, angle = dists_and_angles[min_index]

    return dist, angle


def main():
    with open('./dataset.json') as f:
        dataset = json.load(f)

    classes = dataset['classes']
    model = load_model(classes)

    win = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        # Run MediaPipe Hands.
        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:

            while cap.isOpened():
                cap_ok, frame = cap.read()
                if not cap_ok:
                    continue
                ret_frame, landmarks = run_hands(frame, hands)

                to_show = cv2.flip(
                    frame, 1) if ret_frame is None else ret_frame

                im_center = (
                    int(to_show.shape[1] / 2), int(to_show.shape[0] / 2))
                to_show = cv2.circle(to_show,
                                     im_center,
                                     radius=3,
                                     color=(0, 0, 255),
                                     thickness=3)

                classified_hands = run_inference(classes, landmarks, model)
                classified_hands = ", ".join(classified_hands)

                dist, angle = get_robot_movement(to_show.shape, landmarks)
                if (dist, angle) != (None, None):
                    classified_hands += f" | {dist:.2f}, {angle:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (20, 450)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(to_show, classified_hands, bottomLeftCornerOfText, font,
                            fontScale,
                            fontColor,
                            lineType)

                cv2.imshow(WINDOW_NAME, to_show)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
