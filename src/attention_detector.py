from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EyeMetrics:
    ear: float
    gaze_x: float
    gaze_y: float


@dataclass
class AttentionFeatures:
    face_detected: bool
    left_eye: EyeMetrics | None
    right_eye: EyeMetrics | None
    yaw_proxy: float | None
    pitch_proxy: float | None


class AttentionDetector:
    def __init__(self) -> None:
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    def close(self) -> None:
        return None

    def _estimate_pupil_center(self, eye_roi_gray: np.ndarray) -> tuple[float, float]:
        blurred = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(thresh)
        h, w = eye_roi_gray.shape[:2]
        if moments["m00"] > 1e-5:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = w / 2.0
            cy = h / 2.0
        return cx, cy

    def _eye_metrics_from_box(self, gray_face: np.ndarray, ex: int, ey: int, ew: int, eh: int) -> EyeMetrics:
        eye_roi = gray_face[ey : ey + eh, ex : ex + ew]
        px, py = self._estimate_pupil_center(eye_roi)
        ear = float(eh / max(ew, 1))
        gaze_x = float((px - (ew / 2.0)) / max(ew, 1))
        gaze_y = float((py - (eh / 2.0)) / max(eh, 1))
        return EyeMetrics(ear=ear, gaze_x=gaze_x, gaze_y=gaze_y)

    def process(self, frame_bgr: np.ndarray) -> tuple[AttentionFeatures, np.ndarray]:
        annotated = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
        )

        if len(faces) == 0:
            return (
                AttentionFeatures(
                    face_detected=False,
                    left_eye=None,
                    right_eye=None,
                    yaw_proxy=None,
                    pitch_proxy=None,
                ),
                annotated,
            )

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 128, 0), 2)

        face_gray = gray[y : y + h, x : x + w]
        upper_face_gray = face_gray[0 : int(h * 0.65), :]
        eyes = self._eye_cascade.detectMultiScale(
            upper_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        eyes = sorted(eyes, key=lambda b: b[0])[:2]

        left_eye_metrics = None
        right_eye_metrics = None
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(annotated, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)
            metrics = self._eye_metrics_from_box(upper_face_gray, ex, ey, ew, eh)
            if i == 0:
                left_eye_metrics = metrics
            else:
                right_eye_metrics = metrics

        frame_h, frame_w = frame_bgr.shape[:2]
        face_center_x = x + (w / 2.0)
        face_center_y = y + (h / 2.0)
        yaw_proxy = float((face_center_x - (frame_w / 2.0)) / max(w, 1))
        pitch_proxy = float((face_center_y - (frame_h / 2.0)) / max(h, 1))

        return (
            AttentionFeatures(
                face_detected=True,
                left_eye=left_eye_metrics,
                right_eye=right_eye_metrics,
                yaw_proxy=yaw_proxy,
                pitch_proxy=pitch_proxy,
            ),
            annotated,
        )
