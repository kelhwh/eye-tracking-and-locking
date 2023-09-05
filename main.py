import cv2 as cv
import numpy as np
import mediapipe as mp
import json
from util import add_text
import matplotlib.pyplot as plt
import time


with open('config/landmarks.json') as file:
    landmarks = json.load(file)

#Read mesh points from json
LEFT_IRIS = landmarks['iris']['left']
LEFT_PUPIL = landmarks['pupil']['left']
RIGHT_IRIS = landmarks['iris']['right']
RIGHT_PUPIL = landmarks['pupil']['right']

#The diameter for the central pupil locking area as the percentage of the iris66 diameter
LOCK_SIZE = 1.0

mp_face_mesh = mp.solutions.face_mesh
cap = cv.VideoCapture(0)
plt.ion()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, #Turn on iris and lip mesh points
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1) #Flip vertically to show webcam mirrored image
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        faces = face_mesh.process(frame)

        if faces.multi_face_landmarks: #If face detected in the frame
            landmark = faces.multi_face_landmarks[0].landmark
            # mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmark])
            mesh_points = np.array([[int(p.x * img_w),int(p.y * img_h)] for p in landmark])

            #Draw both irises and create detection status
            cv.circle(frame, mesh_points[LEFT_PUPIL], np.linalg.norm(mesh_points[LEFT_PUPIL] - mesh_points[LEFT_IRIS][0]).astype(int), (255, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_PUPIL], radius=1, color=(255, 0, 255), thickness=-1)
            cv.circle(frame, mesh_points[RIGHT_PUPIL], np.linalg.norm(mesh_points[RIGHT_PUPIL] - mesh_points[RIGHT_IRIS][0]).astype(int), (255, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_PUPIL], radius=1, color=(255, 0, 255), thickness=-1)

            text_detect = f"[Detection]\nLeft eye: {mesh_points[LEFT_PUPIL]} \nRight eye: {mesh_points[RIGHT_PUPIL]}"

        else:
            text_detect = f"[Detection]\nLeft eye: Not detected\nRight eye: Not detected"


        #Draw locking range and create locking status
        if 'lock_left_pupil' in globals():
            deviation_left = np.linalg.norm(mesh_points[LEFT_PUPIL] - lock_left_pupil).round(1)
            is_deviated_left = deviation_left > lock_left_diameter

            text_lock = f"[Lock]\nLeft eye: Locked {lock_left_pupil}  Deviation: {deviation_left}"
            lock_color = (0,255,0)

            if is_deviated_left:
                text_lock += "\nWARNING"
                lock_color = (0,0,255)
            cv.circle(frame, lock_left_pupil, lock_left_diameter, lock_color, thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, lock_left_pupil, radius=1, color=lock_color, thickness=-1)

        else:
            text_lock = f"\n Left eye: Not locked  Deviation: Not locked"
            lock_color = (0,255,0)

        #Show text for status
        add_text(
            frame,
            text_detect,
            org=(int(img_w*0.05), int(img_h*0.1)),
            font=cv.FONT_HERSHEY_COMPLEX,
            font_scale=0.5,
            color=(0,255,0)
        )
        add_text(
            frame,
            text_lock,
            org=(int(img_w*0.3), int(img_h*0.1)),
            font=cv.FONT_HERSHEY_COMPLEX,
            font_scale=0.5,
            color=lock_color
        )

        cv.imshow('img', frame)
        if 't_start' in globals():

            t=round(time.time() - t_start,2)
            time_record = np.append(time_record, t)
            deviation_record = np.append(deviation_record, deviation_left)


        #Reading keyboard input
        key = cv.waitKey(1)
        if key == ord('q'): #q for quit
            break
        elif key == ord('l'):#l for locking left eye
            if faces.multi_face_landmarks:
                lock_left_pupil = mesh_points[LEFT_PUPIL]
                lock_left_diameter = int(np.linalg.norm(mesh_points[LEFT_PUPIL] - mesh_points[LEFT_IRIS][0]) * LOCK_SIZE)
                t_start = time.time()
                deviation_record = np.array(0)
                time_record = np.array(0)
        elif key == 12: #crtl+l for unlocking left eye
            try:
                del lock_left_pupil
                del lock_left_diameter
            except:
                continue


cap.release()
cv.destroyAllWindows()
