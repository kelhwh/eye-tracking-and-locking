import cv2 as cv
import numpy as np
import mediapipe as mp
import json

with open('landmarks.json') as file:
    landmarks = json.load(file)

LEFT_IRIS = landmarks['iris']['left']
LEFT_PUPIL = landmarks['pupil']['left']
RIGHT_IRIS = landmarks['iris']['right']
RIGHT_PUPIL = landmarks['pupil']['right']

mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1) #Flip vertically for webcam mirrored image
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        faces = face_mesh.process(frame)

        if faces.multi_face_landmarks:
            landmark = faces.multi_face_landmarks[0].landmark
            # mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmark])
            mesh_points = np.array([[int(p.x * img_w),int(p.y * img_h)] for p in landmark])

            cv.circle(frame, mesh_points[LEFT_PUPIL], np.linalg.norm(mesh_points[LEFT_PUPIL] - mesh_points[LEFT_IRIS][0]).astype(int), (0, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_PUPIL], radius=1, color=(0, 0, 255), thickness=-1)
            cv.circle(frame, mesh_points[RIGHT_PUPIL], np.linalg.norm(mesh_points[RIGHT_PUPIL] - mesh_points[RIGHT_IRIS][0]).astype(int), (0, 0, 255), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_PUPIL], radius=1, color=(0, 0, 255), thickness=-1)

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
