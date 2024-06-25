#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

model_path = 'face_landmarker_v2_with_blendshapes.task'

from utils import CvFpsCalc
from model import StaticClassifier
from model import FaceGestureClassifier

SELECTED_STATIC = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
SELECTED_DYN = [10, 168, 6, 94, 0, 17, 152, 162, 127, 389, 356 ]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_brect = True

    # Camera start with config ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe model load #############################################################
    mp_face_mesh = mp.solutions.face_mesh

    mp_face = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence)

    static_classifier = StaticClassifier()
    gesture_classifier = FaceGestureClassifier()

    # Read labels ###########################################################
    with open('model/static_classifier/static_classifier_label.csv',
              encoding='utf-8-sig') as f:
        static_classifier_labels = csv.reader(f)
        static_classifier_labels = [
            row[0] for row in static_classifier_labels
        ]
    with open(
            'model/gesture_classifier/gesture_label.csv',
            encoding='utf-8-sig') as f:
        gesture_classifier_labels = csv.reader(f)
        gesture_classifier_labels = [
            row[0] for row in gesture_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Face gesture time frame (depends on fps, adjust accordingly)
    history_length = 32
    gesture = deque(maxlen=history_length)

    # Starting mode ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: stop) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            print('Camera failed to start')
            break
        image = cv.flip(image, 1)  # Mirror display
        edit_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = mp_face.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        gesture_pause = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #face tesselation grid
                # mp_drawing.draw_landmarks(
                #     image=edit_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())
                
                #face, eyes, eyebrows contour
                # mp.solutions.drawing_utils.draw_landmarks(
                #     image=edit_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp.solutions.drawing_styles
                #     .get_default_face_mesh_contours_style())
                
                
                #irises diamond tracking
                # mp_drawing.draw_landmarks(
                #     image=edit_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_iris_connections_style())

                # Bounding box calculation
                brect = calc_bounding_rect(edit_image, face_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(edit_image, face_landmarks)

                # filter-out face contours points
                selected_static_list = []
                selected_dyn_list = []
                for i in range(len(landmark_list)):
                    if i in SELECTED_STATIC:
                        selected_static_list += [landmark_list[i]]
                    if i in SELECTED_DYN:
                        selected_dyn_list += [landmark_list[i]]

                # Conversion to relative coordinates / normalized coordinates
                prep_landmark_list = pre_process_landmark(selected_static_list)

                gesture.append(selected_dyn_list)
                prep_gesture_list = pre_process_gesture(edit_image, gesture)

                face_label = static_classifier(prep_landmark_list)

                #default to no gesture
                face_gesture_label = 0
                if len(gesture) >= 32 and gesture_pause <= 0 and mode == 3:
                    face_gesture_label = gesture_classifier(prep_gesture_list)

                # Write to the dataset file
                if mode == 1 or mode == 2:
                    logging_csv(number, mode, prep_landmark_list, prep_gesture_list)

                if mode != 3:
                    edit_image = draw_info_text(edit_image, brect, static_classifier_labels[face_label])

                # Some gesture detected
                if face_gesture_label != 0:
                    gesture_pause = 32
                    print(f'Gesture detected: {gesture_classifier_labels[face_gesture_label]}')

                gesture_pause = max(0, gesture_pause-1)
                
        edit_image = draw_info(edit_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Face Gesture Recognition', edit_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110 or key == 78:  # n or N static inference
        mode = 0
    if key == 107 or key == 75:  # k or K static record
        mode = 1
    if key == 108 or key == 76:  # l or L gesture record
        mode = 2
    if key == 120 or key == 88:  #x or X gesture inference
        mode = 3

    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # static
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_gesture(image, gesture):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_gesture = copy.deepcopy(gesture)

    res_list = []

    # Convert to relative coordinates with respect to the starting position
    base_x, base_y = gesture[0][0][0], gesture[0][0][1]
    for list_ele in temp_gesture:
        norm_list_ele = []
        for i in range(len(list_ele)):
            norm_list_ele += [(list_ele[i][0]-base_x)/image_width,(list_ele[i][1]-base_y)/image_height]
        res_list += norm_list_ele

    # Convert to a one-dimensional list
    # temp_gesture = list(itertools.chain.from_iterable(temp_gesture))

    return res_list


def logging_csv(number, mode, landmark_list, gesture_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/static_classifier/static_face.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/gesture_classifier/gesture_face.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *gesture_list])


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


# def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
def draw_info_text(image, brect, face_label):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = 'Action: ' + face_label
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History', 'Gesture Inference']
    if 1 <= mode <= 3:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
