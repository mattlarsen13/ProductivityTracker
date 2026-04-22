from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from attention_detector import AttentionFeatures

AttentionState = Literal["focused", "brief_away", "distracted"]


@dataclass
class ThresholdConfig:
    yaw_threshold: float = 0.45
    pitch_threshold: float = 0.25
    gaze_x_threshold: float = 0.22
    gaze_y_threshold: float = 0.28
    missing_face_seconds: float = 1.2
    brief_away_seconds: float = 0.8
    distracted_seconds: float = 2.5

    def reset_defaults(self) -> None:
        self.yaw_threshold = 0.45
        self.pitch_threshold = 0.25
        self.gaze_x_threshold = 0.22
        self.gaze_y_threshold = 0.28
        self.missing_face_seconds = 1.2
        self.brief_away_seconds = 0.8
        self.distracted_seconds = 2.5


@dataclass
class ClassificationResult:
    state: AttentionState
    reason: str | None
    away_seconds: float
    no_face_seconds: float
    avg_gaze_x: float | None
    avg_gaze_y: float | None


class DistractionClassifier:
    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()
        self._away_started_at: float | None = None
        self._no_face_started_at: float | None = None
        self._state: AttentionState = "focused"

    @property
    def state(self) -> AttentionState:
        return self._state

    def _average_gaze(self, features: AttentionFeatures) -> tuple[float | None, float | None]:
        gaze_x_values: list[float] = []
        gaze_y_values: list[float] = []
        if features.left_eye is not None:
            gaze_x_values.append(features.left_eye.gaze_x)
            gaze_y_values.append(features.left_eye.gaze_y)
        if features.right_eye is not None:
            gaze_x_values.append(features.right_eye.gaze_x)
            gaze_y_values.append(features.right_eye.gaze_y)
        if not gaze_x_values:
            return None, None
        return (sum(gaze_x_values) / len(gaze_x_values), sum(gaze_y_values) / len(gaze_y_values))

    def update(self, features: AttentionFeatures, now: float) -> ClassificationResult:
        reason: str | None = None
        is_away = False
        no_face_seconds = 0.0
        avg_gaze_x, avg_gaze_y = self._average_gaze(features)

        if not features.face_detected:
            if self._no_face_started_at is None:
                self._no_face_started_at = now
            no_face_seconds = now - self._no_face_started_at
            if no_face_seconds >= self.config.missing_face_seconds:
                is_away = True
                reason = "no_face"
        else:
            self._no_face_started_at = None
            if features.yaw_proxy is not None and abs(features.yaw_proxy) >= self.config.yaw_threshold:
                is_away = True
                reason = "head_yaw"
            elif features.pitch_proxy is not None and abs(features.pitch_proxy) >= self.config.pitch_threshold:
                is_away = True
                reason = "head_pitch"
            elif avg_gaze_x is not None and abs(avg_gaze_x) >= self.config.gaze_x_threshold:
                is_away = True
                reason = "gaze_x"
            elif avg_gaze_y is not None and abs(avg_gaze_y) >= self.config.gaze_y_threshold:
                is_away = True
                reason = "gaze_y"

        away_seconds = 0.0
        if is_away:
            if self._away_started_at is None:
                self._away_started_at = now
            away_seconds = now - self._away_started_at
            if away_seconds >= self.config.distracted_seconds:
                self._state = "distracted"
            elif away_seconds >= self.config.brief_away_seconds:
                self._state = "brief_away"
            else:
                self._state = "focused"
        else:
            self._away_started_at = None
            self._state = "focused"

        return ClassificationResult(
            state=self._state,
            reason=reason,
            away_seconds=away_seconds,
            no_face_seconds=no_face_seconds,
            avg_gaze_x=avg_gaze_x,
            avg_gaze_y=avg_gaze_y,
        )
