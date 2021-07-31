import json
import sys

import cv2
import mediapipe as mp
import torch

from training.model import HandsClassifier

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

WINDOW_NAME = "win"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


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

    return cv2.flip(annotated_image, 1), results.multi_hand_landmarks


def format_landmarks(landmarks):
    if landmarks is None:
        return None

    ret = []
    for landmark in landmarks:
        f_landmarks = [[point.x, point.y] for point in landmark.landmark]
        ret.append(torch.tensor([f_landmarks, ]))

    return ret


def class_as_str(classes, class_index):
    return classes[class_index]['name']


def main():
    with open('./dataset.json') as f:
        dataset = json.load(f)

    classes = dataset['classes']
    n_classes = len(classes)

    model = HandsClassifier(27)
    model.load_state_dict(torch.load('./output/best.pt'))
    model.eval()

    win = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")


    cap_ok, frame = cap.read()
    print(frame.shape)

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

                f_landmarks = format_landmarks(landmarks)
                classified_hands = []
                if not f_landmarks is None:
                    for landmark in f_landmarks:
                        class_index = torch.argmax(model(landmark)).item()
                        classified_hands.append(
                            class_as_str(classes, class_index))

                classified_hands = ", ".join(classified_hands)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (20, 450)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2

                to_show = frame if ret_frame is None else ret_frame
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
