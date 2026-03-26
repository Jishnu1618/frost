import cv2
import numpy as np

# Load Haar cascade for eye detection (ships with OpenCV)
_eye_cascade = None

def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        _eye_cascade = cv2.CascadeClassifier(cascade_path)
    return _eye_cascade


def _estimate_keypoints(x1, y1, x2, y2):
    """
    Estimate skeletal keypoints from a bounding box using anatomical proportions.
    Returns dict of keypoint names -> (x, y) coordinates.
    """
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2  # Center X

    return {
        "head":           (cx, y1 + int(h * 0.08)),
        "neck":           (cx, y1 + int(h * 0.18)),
        "left_shoulder":  (x1 + int(w * 0.25), y1 + int(h * 0.22)),
        "right_shoulder": (x1 + int(w * 0.75), y1 + int(h * 0.22)),
        "left_elbow":     (x1 + int(w * 0.15), y1 + int(h * 0.38)),
        "right_elbow":    (x1 + int(w * 0.85), y1 + int(h * 0.38)),
        "left_wrist":     (x1 + int(w * 0.18), y1 + int(h * 0.52)),
        "right_wrist":    (x1 + int(w * 0.82), y1 + int(h * 0.52)),
        "left_hip":       (x1 + int(w * 0.35), y1 + int(h * 0.52)),
        "right_hip":      (x1 + int(w * 0.65), y1 + int(h * 0.52)),
        "left_knee":      (x1 + int(w * 0.32), y1 + int(h * 0.72)),
        "right_knee":     (x1 + int(w * 0.68), y1 + int(h * 0.72)),
        "left_ankle":     (x1 + int(w * 0.30), y1 + int(h * 0.95)),
        "right_ankle":    (x1 + int(w * 0.70), y1 + int(h * 0.95)),
    }


SKELETON_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]

STICK_COLOR = (255, 229, 0)
JOINT_COLOR = (255, 150, 0)
HEAD_COLOR  = (255, 229, 0)
BONE_THICKNESS = 2
JOINT_RADIUS = 4
HEAD_RADIUS_RATIO = 0.08


def _draw_stick_figure(frame, x1, y1, x2, y2):
    """Draw a neon-styled stick figure over a person bounding box."""
    kp = _estimate_keypoints(x1, y1, x2, y2)
    h = y2 - y1

    for (a, b) in SKELETON_CONNECTIONS:
        cv2.line(frame, kp[a], kp[b], STICK_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    for name, pt in kp.items():
        if name != "head":
            cv2.circle(frame, pt, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

    head_radius = max(int(h * HEAD_RADIUS_RATIO), 8)
    cv2.circle(frame, kp["head"], head_radius, HEAD_COLOR, 2, cv2.LINE_AA)


def _blur_eyes_and_face(frame, x1, y1, x2, y2):
    """
    Two-pass privacy blur:
    1. Try Haar cascade to find actual eyes and blur a band across them.
    2. Always apply a fallback proportional eye-band blur (top 8-20% of bbox)
       to guarantee coverage even when Haar misses.
    """
    fh, fw = frame.shape[:2]
    h = y2 - y1
    w = x2 - x1
    if h < 20 or w < 10:
        return  # Too small to process

    # --- Pass 1: Haar cascade eye detection on head region ---
    head_y1 = max(y1, 0)
    head_y2 = min(y1 + int(h * 0.35), fh)
    head_x1 = max(x1, 0)
    head_x2 = min(x2, fw)

    if head_y2 > head_y1 and head_x2 > head_x1:
        head_roi = frame[head_y1:head_y2, head_x1:head_x2]
        if head_roi.size > 0:
            cascade = _get_eye_cascade()
            gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            eyes = cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            
            if len(eyes) > 0:
                # Blur each detected eye with generous padding
                for (ex, ey, ew, eh) in eyes:
                    pad_x = int(ew * 0.4)
                    pad_y = int(eh * 0.5)
                    ey1 = max(ey - pad_y, 0)
                    ey2 = min(ey + eh + pad_y, head_roi.shape[0])
                    ex1 = max(ex - pad_x, 0)
                    ex2 = min(ex + ew + pad_x, head_roi.shape[1])
                    eye_region = head_roi[ey1:ey2, ex1:ex2]
                    if eye_region.size > 0:
                        k = max(31, ((ey2 - ey1) * 2) | 1)
                        head_roi[ey1:ey2, ex1:ex2] = cv2.GaussianBlur(eye_region, (k, k), 20)

    # --- Pass 2: Always apply proportional eye-band blur as guaranteed fallback ---
    # Eye band: approximately 5-18% from top of bbox 
    band_y1 = max(y1 + int(h * 0.05), 0)
    band_y2 = min(y1 + int(h * 0.20), fh)
    band_x1 = max(x1 + int(w * 0.10), 0)
    band_x2 = min(x2 - int(w * 0.10), fw)

    if band_y2 > band_y1 and band_x2 > band_x1:
        band = frame[band_y1:band_y2, band_x1:band_x2]
        if band.size > 0:
            k = max(45, (band_y2 - band_y1) | 1)
            frame[band_y1:band_y2, band_x1:band_x2] = cv2.GaussianBlur(band, (k, k), 25)


def apply_ghost_mode(frame, detections):
    '''
    Privacy-first anonymization: blurs eyes/face and overlays stick figures.
    Operates in-place on a copy for performance.
    detections: list of tuples (x1, y1, x2, y2, confidence)
    '''
    anonymized_frame = frame.copy()
    
    for (x1, y1, x2, y2, conf) in detections:
        _blur_eyes_and_face(anonymized_frame, x1, y1, x2, y2)
        _draw_stick_figure(anonymized_frame, x1, y1, x2, y2)
    
    return anonymized_frame
