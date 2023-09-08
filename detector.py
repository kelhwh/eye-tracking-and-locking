import cv2 as cv
import numpy as np
import mediapipe as mp
import json
from util import add_text
import matplotlib.pyplot as plt
import time


class Detector():
    def __init__(self):
        with open('config/landmarks.json') as file:
            #Read mesh points from json
            landmarks = json.load(file)
            self.landmark_map = landmarks

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, #Turn on iris and lip mesh points
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )

        self.pupil_left = None
        self.pupil_right = None
        self.iris_left = None
        self.iris_right = None
        self.pupil_diameter = None
        self.deviation_left = None
        self.is_deviated_left = None

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        faces = self.face_mesh.process(frame)

        key_landmarks = None
        diameter = None
        #If face detected in the frame
        self.face_detected = True if faces.multi_face_landmarks else False

        if self.face_detected:
            landmark = faces.multi_face_landmarks[0].landmark
            # mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmark])
            mesh_points = np.array([[int(p.x * img_w),int(p.y * img_h)] for p in landmark])
            self.pupil_left = mesh_points[self.landmark_map['pupil']['left']]
            self.pupil_right = mesh_points[self.landmark_map['pupil']['right']]
            self.iris_left = mesh_points[self.landmark_map['iris']['left']]
            self.iris_right = mesh_points[self.landmark_map['iris']['right']]

            self.pupil_diameter = np.linalg.norm(self.pupil_left - self.iris_left[0]).astype(int)


    def label(self, frame, lock_status, lock_point, lock_diameter):
        img_h, img_w = frame.shape[:2]

        if self.face_detected:
            #Draw both irises and pupil and create detection status text
            cv.circle(frame, self.pupil_left, self.pupil_diameter, (255, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, self.pupil_left, radius=1, color=(255, 0, 255), thickness=-1)
            cv.circle(frame, self.pupil_right, self.pupil_diameter, (255, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, self.pupil_right, radius=1, color=(255, 0, 255), thickness=-1)

            text_detect = f"[Detection]\nLeft eye: {self.pupil_left} \nRight eye: {self.pupil_right}"

        else:
            text_detect = f"[Detection]\nLeft eye: Not detected\nRight eye: Not detected"

        #Draw locking range and create locking status
        if lock_status:
            lock_left_pupil = lock_point
            lock_left_diameter = lock_diameter
            text_lock = f"[Lock]\nLeft eye: Locked {lock_left_pupil}"
            lock_color = (0,255,0)

            if self.face_detected:
                self.deviation_left = np.linalg.norm(self.pupil_left - lock_left_pupil).round(1)
                self.is_deviated_left = self.deviation_left > lock_left_diameter

                text_lock += f"  Deviation: {self.deviation_left}"

                if self.is_deviated_left:
                    text_lock += "\nWARNING"
                    lock_color = (255,0,0)

            else:
                text_lock += "\nWARNING: Face not detected!"
                lock_color = (255,0,0)

            cv.circle(frame, lock_left_pupil, lock_left_diameter, lock_color, thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, lock_left_pupil, radius=1, color=lock_color, thickness=-1)

        else:
            text_lock = f"\n Left eye: Not locked  Deviation: Not locked"
            lock_color = (0,255,0)


        add_text(
            frame,
            text_detect,
            org=(int(img_w*0.05), int(img_h*0.1)),
            font=cv.FONT_HERSHEY_TRIPLEX,
            font_scale=0.9,
            color=(0,255,0)
        )
        add_text(
            frame,
            text_lock,
            org=(int(img_w*0.3), int(img_h*0.1)),
            font=cv.FONT_HERSHEY_TRIPLEX,
            font_scale=0.9,
            color=lock_color
        )

        return frame
