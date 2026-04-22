from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera index {self.camera_index}.")

    def read(self):
        if self._cap is None:
            raise CameraError("Camera not opened.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera.")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
