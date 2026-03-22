import dlib
import cv2
from math import hypot

predictorModel = 'models/shape_predictor_model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictorModel)


def calcDistance(pointA, pointB):
    return hypot(pointA[0] - pointB[0], pointA[1] - pointB[1])


def mouthTrack(faces, frame):
    """
    Returns "Mouth Open" or "Mouth Close".

    FIX: The original used a fixed pixel threshold of 23px which fires
    constantly during normal speech.  We now compute the threshold as a
    fraction of the inter-eye distance so it scales with face size and
    camera distance, and we raise the open/close ratio to only flag
    genuinely wide-open mouths (reading aloud / whispering), not normal
    talking.

    Ratio used:  mouth_opening / inter_eye_width
    Threshold:   > 0.35  →  Mouth Open   (was a fixed 23px)
    """
    for face in faces:
        lm = predictor(frame, face)

        # Mouth vertical opening (outer lip top → bottom)
        mouth_top    = (lm.part(51).x, lm.part(51).y)
        mouth_bottom = (lm.part(57).x, lm.part(57).y)
        opening = calcDistance(mouth_top, mouth_bottom)

        # Inter-eye width as normalisation reference
        left_eye_corner  = (lm.part(36).x, lm.part(36).y)
        right_eye_corner = (lm.part(45).x, lm.part(45).y)
        eye_width = max(calcDistance(left_eye_corner, right_eye_corner), 1.0)

        ratio = opening / eye_width

        # > 0.35 = wide open (shouting, whispering to someone)
        # 0.10–0.35 = normal speech — not flagged
        if ratio > 0.35:
            return "Mouth Open"
        else:
            return "Mouth Close"

    return "Mouth Close"