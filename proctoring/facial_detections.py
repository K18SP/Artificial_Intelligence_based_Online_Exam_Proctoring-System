# This will detect the face

import dlib
import cv2
from imutils import face_utils


# -------- Load models once -------- #

shapePredictorModel = 'models/shape_predictor_model/shape_predictor_68_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(shapePredictorModel)


def detectFace(frame):
    """
    Input: video frame from webcam
    Output: face count and detected faces
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector(gray, 0)

    faceCount = len(faces)

    for face in faces:

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # ---------- Fancy bounding box ---------- #

        cv2.line(frame, (x, y), (x + 20, y), (0,255,255), 2)
        cv2.line(frame, (x, y), (x, y + 20), (0,255,255), 2)

        cv2.line(frame, (x + w, y), (x + w - 20, y), (0,255,255), 2)
        cv2.line(frame, (x + w, y), (x + w, y + 20), (0,255,255), 2)

        cv2.line(frame, (x, y + h), (x + 20, y + h), (0,255,255), 2)
        cv2.line(frame, (x, y + h), (x, y + h - 20), (0,255,255), 2)

        cv2.line(frame, (x + w, y + h), (x + w - 20, y + h), (0,255,255), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - 20), (0,255,255), 2)

        # ---------- Landmarks ---------- #

        facialLandmarks = shapePredictor(gray, face)
        facialLandmarks = face_utils.shape_to_np(facialLandmarks)

        for (a, b) in facialLandmarks:
            cv2.circle(frame, (a, b), 2, (255,255,0), -1)

    return faceCount, faces