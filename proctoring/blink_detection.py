import dlib
from math import hypot
import cv2

shapePredictorModel = 'models/shape_predictor_model/shape_predictor_68_face_landmarks.dat'
shapePredictor = dlib.shape_predictor(shapePredictorModel)

# ── Debounce state ────────────────────────────────────────────────────────────
# A blink is counted once when the eye CLOSES (ratio crosses threshold),
# then ignored until the eye OPENS again.  Prevents 20+ counts per blink.
_was_blinking = False


def midPoint(pointA, pointB):
    # FIX: original code did int(x)/2 which caused integer truncation BEFORE
    # the division — vertical distances came out wrong, making EAR too high.
    # Correct: compute float average of both coords.
    X = (pointA.x + pointB.x) / 2.0
    Y = (pointA.y + pointB.y) / 2.0
    return (X, Y)


def findDist(pointA, pointB):
    return hypot(pointA[0] - pointB[0], pointA[1] - pointB[1])


def isBlinking(faces, frame):
    """
    Returns (lRatio, rRatio, status) where status is "Blink" or "No Blink".

    "Blink" fires only on the FALLING EDGE of eye closure (i.e. the first
    frame where both eyes cross the EAR threshold after being open).
    This means the caller's blinkCount increments at most once per blink
    regardless of how many frames the eyes stay shut.

    EAR threshold tuned to 4.5 (was 3.6 — too sensitive, fired constantly).
    Require BOTH eyes to blink to avoid false positives from head tilt.
    """
    global _was_blinking

    ratio = ()

    for face in faces:
        lm = shapePredictor(frame, face)

        # ── Left eye ──
        lL = (lm.part(36).x, lm.part(36).y)
        lR = (lm.part(39).x, lm.part(39).y)
        lT = midPoint(lm.part(37), lm.part(38))
        lB = midPoint(lm.part(40), lm.part(41))
        lRatio = findDist(lL, lR) / max(findDist(lT, lB), 0.1)

        # ── Right eye ──
        rL = (lm.part(42).x, lm.part(42).y)
        rR = (lm.part(45).x, lm.part(45).y)
        rT = midPoint(lm.part(43), lm.part(44))
        rB = midPoint(lm.part(46), lm.part(47))
        rRatio = findDist(rL, rR) / max(findDist(rT, rB), 0.1)

        # ── Both eyes must close to count as a blink ──
        THRESHOLD = 4.5
        both_closed = (lRatio >= THRESHOLD and rRatio >= THRESHOLD)

        if both_closed and not _was_blinking:
            # Falling edge — genuine new blink
            _was_blinking = True
            ratio += (lRatio, rRatio, "Blink")
        elif not both_closed:
            # Eyes open — reset so next closure is a new blink
            _was_blinking = False
            ratio += (lRatio, rRatio, "No Blink")
        else:
            # Eyes still closed from previous frame — do NOT count again
            ratio += (lRatio, rRatio, "No Blink")

    return ratio