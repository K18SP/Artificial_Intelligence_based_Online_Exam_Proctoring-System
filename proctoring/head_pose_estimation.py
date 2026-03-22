import numpy as np
import dlib
import math
import cv2

shapePredictorModel = 'models/shape_predictor_model/shape_predictor_68_face_landmarks.dat'
shapePredictor = dlib.shape_predictor(shapePredictorModel)

# ── 3-D face model points (generic human skull) ──────────────────────────────
model_points = np.array([
    ( 0.0,    0.0,    0.0),      # Nose tip        (30)
    ( 0.0, -330.0,  -65.0),      # Chin            (8)
    (-225.0,  170.0, -135.0),    # Left eye corner (36)
    ( 225.0,  170.0, -135.0),    # Right eye corner(45)
    (-150.0, -150.0, -125.0),    # Left mouth      (48)
    ( 150.0, -150.0, -125.0),    # Right mouth     (54)
], dtype=np.float64)

# ── Calibration: baseline angles measured over first N frames ─────────────────
# The head is assumed to be roughly centred at the start of the session.
# We average ang1/ang2 over CALIB_FRAMES and subtract that offset from every
# subsequent reading.  This removes the constant "Head Right" bias caused by
# a slightly off-axis camera or asymmetric face geometry.

CALIB_FRAMES   = 30          # ~1 second at 30 fps
_calib_ang1    = []
_calib_ang2    = []
_offset_ang1   = 0.0
_offset_ang2   = 0.0
_calibrated    = False


def _calibrate(ang1: float, ang2: float):
    global _calibrated, _offset_ang1, _offset_ang2
    _calib_ang1.append(ang1)
    _calib_ang2.append(ang2)
    if len(_calib_ang1) >= CALIB_FRAMES:
        _offset_ang1 = float(np.mean(_calib_ang1))
        _offset_ang2 = float(np.mean(_calib_ang2))
        _calibrated  = True
        print(f"Head pose calibrated — offset: ang1={_offset_ang1:.1f}° ang2={_offset_ang2:.1f}°")


def _compute_angles(img, marks):
    """Return (ang1_vertical, ang2_horizontal) in degrees."""
    image_points = np.array([
        [marks.part(30).x, marks.part(30).y],
        [marks.part(8).x,  marks.part(8).y ],
        [marks.part(36).x, marks.part(36).y],
        [marks.part(45).x, marks.part(45).y],
        [marks.part(48).x, marks.part(48).y],
        [marks.part(54).x, marks.part(54).y],
    ], dtype=np.float64)

    h, w = img.shape[:2]
    focal  = w                       # approximate focal length
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_UPNP
    )
    if not ok:
        return 0.0, 0.0

    # Project nose tip forward to get direction vector
    nose_3d  = np.array([[0.0, 0.0, 1000.0]])
    nose_2d, _ = cv2.projectPoints(nose_3d, rvec, tvec, cam_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_2d[0][0][0]),   int(nose_2d[0][0][1]))

    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    ang1 = math.degrees(math.atan2(dy1, dx1)) if dx1 != 0 else 90.0

    # Horizontal reference from projected head box
    rear_size, front_size = 1, w
    pt3d = np.array([
        (-rear_size,  -rear_size,  0),
        (-rear_size,   rear_size,  0),
        ( front_size,  front_size, front_size * 2),
        ( front_size, -front_size, front_size * 2),
        (-rear_size,  -rear_size,  0),
    ], dtype=np.float64)
    pts2d, _ = cv2.projectPoints(pt3d, rvec, tvec, cam_matrix, dist_coeffs)
    pts2d = np.int32(pts2d.reshape(-1, 2))
    y_ref = (pts2d[0] + pts2d[3]) // 2
    x_ref = pts2d[2]

    dx2 = int(x_ref[0]) - int(y_ref[0])
    dy2 = int(x_ref[1]) - int(y_ref[1])
    ang2 = math.degrees(math.atan2(dy2, dx2)) if dx2 != 0 else 90.0

    return ang1, ang2


def head_pose_detection(faces, img):
    """
    Returns one of: "Head Up", "Head Down", "Head Left", "Head Right",
    "Center", or -1 (center, no label needed).

    FIX: runtime calibration removes the constant directional bias caused
    by an asymmetrically positioned camera or face geometry.
    Threshold raised to 15° (was 45°) after calibration offset is applied,
    keeping sensitivity for real lateral movement.
    """
    global _calibrated

    for face in faces:
        try:
            marks = shapePredictor(img, face)
            ang1, ang2 = _compute_angles(img, marks)
        except Exception as e:
            print(f"head_pose_detection compute error: {e}")
            return "Center"

        if not _calibrated:
            _calibrate(ang1, ang2)
            return "Center"    # don't classify during calibration

        # Apply calibration offset
        adj1 = ang1 - _offset_ang1
        adj2 = ang2 - _offset_ang2

        THRESHOLD = 15.0      # degrees — tighter than original 45°

        if adj1 >= THRESHOLD:
            return "Head Up"
        if adj1 <= -THRESHOLD:
            return "Head Down"
        if adj2 >= THRESHOLD:
            return "Head Right"
        if adj2 <= -THRESHOLD:
            return "Head Left"

        return "Center"

    return "Center"