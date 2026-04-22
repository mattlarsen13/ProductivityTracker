from __future__ import annotations

import argparse
import time

import cv2

from attention_detector import AttentionDetector
from camera import Camera, CameraError
from distraction_state import DistractionClassifier, ThresholdConfig


def _overlay_text(frame, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


def _adjust_thresholds(selected_idx: int, config: ThresholdConfig, direction: int) -> None:
    if selected_idx == 1:
        config.yaw_threshold = max(0.05, min(2.0, config.yaw_threshold + (0.02 * direction)))
    elif selected_idx == 2:
        config.pitch_threshold = max(0.05, min(2.0, config.pitch_threshold + (0.02 * direction)))
    elif selected_idx == 3:
        config.gaze_x_threshold = max(0.02, min(1.0, config.gaze_x_threshold + (0.01 * direction)))
    elif selected_idx == 4:
        config.gaze_y_threshold = max(0.02, min(1.0, config.gaze_y_threshold + (0.01 * direction)))
    elif selected_idx == 5:
        config.missing_face_seconds = max(0.1, min(10.0, config.missing_face_seconds + (0.1 * direction)))
    elif selected_idx == 6:
        config.brief_away_seconds = max(0.1, min(10.0, config.brief_away_seconds + (0.1 * direction)))
        config.distracted_seconds = max(config.brief_away_seconds + 0.1, config.distracted_seconds)
    elif selected_idx == 7:
        config.distracted_seconds = max(
            config.brief_away_seconds + 0.1,
            min(20.0, config.distracted_seconds + (0.1 * direction)),
        )


def run(camera_index: int, no_gui: bool, frames: int) -> int:
    camera = Camera(camera_index=camera_index)
    detector = AttentionDetector()
    config = ThresholdConfig()
    classifier = DistractionClassifier(config=config)
    selected_idx = 1
    try:
        camera.open()
        processed = 0
        state_counts = {"focused": 0, "brief_away": 0, "distracted": 0}
        while True:
            frame = camera.read()
            features, annotated = detector.process(frame)
            now = time.monotonic()
            classification = classifier.update(features, now)
            state_counts[classification.state] += 1
            lines: list[str] = []

            if not features.face_detected:
                lines.append("Face: not detected")
            else:
                lines.append("Face: detected")
                if features.left_eye is None:
                    lines.append("Left eye: not detected")
                else:
                    lines.append(
                        f"Left eye - EAR: {features.left_eye.ear:.3f}, gaze(x,y): ({features.left_eye.gaze_x:.3f}, {features.left_eye.gaze_y:.3f})"
                    )
                if features.right_eye is None:
                    lines.append("Right eye: not detected")
                else:
                    lines.append(
                        f"Right eye - EAR: {features.right_eye.ear:.3f}, gaze(x,y): ({features.right_eye.gaze_x:.3f}, {features.right_eye.gaze_y:.3f})"
                    )
                lines.append(
                    f"Head proxy - yaw: {features.yaw_proxy:.3f}, pitch: {features.pitch_proxy:.3f}"
                )
            lines.append(
                f"State: {classification.state} | reason: {classification.reason or 'none'} | away_s: {classification.away_seconds:.2f}"
            )
            lines.append(
                f"Gaze avg(x,y): ({classification.avg_gaze_x if classification.avg_gaze_x is not None else 0.0:.3f}, {classification.avg_gaze_y if classification.avg_gaze_y is not None else 0.0:.3f}) | no_face_s: {classification.no_face_seconds:.2f}"
            )
            processed += 1

            if no_gui:
                if processed == 1:
                    print("First frame features:")
                    for line in lines:
                        print(f"- {line}")
                if processed >= frames:
                    print(f"Processed {processed} frames (no GUI mode).")
                    print(
                        "State counts: "
                        f"focused={state_counts['focused']}, "
                        f"brief_away={state_counts['brief_away']}, "
                        f"distracted={state_counts['distracted']}"
                    )
                    break
            else:
                lines.append(
                    f"Thresholds [selected={selected_idx}]: 1:yaw={config.yaw_threshold:.2f} 2:pitch={config.pitch_threshold:.2f} 3:gazeX={config.gaze_x_threshold:.2f} 4:gazeY={config.gaze_y_threshold:.2f}"
                )
                lines.append(
                    f"Thresholds: 5:noFaceS={config.missing_face_seconds:.1f} 6:briefS={config.brief_away_seconds:.1f} 7:distractedS={config.distracted_seconds:.1f}"
                )
                lines.append("Controls: [1-7] select metric, Up/Down adjust, R reset, P print, Q quit")
                _overlay_text(annotated, lines)
                cv2.imshow("ProductivityTracker Step 3 Attention Classification", annotated)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7")):
                    selected_idx = int(chr(key))
                elif key == 2490368:
                    _adjust_thresholds(selected_idx, config, direction=1)
                elif key == 2621440:
                    _adjust_thresholds(selected_idx, config, direction=-1)
                elif key in (ord("r"), ord("R")):
                    config.reset_defaults()
                elif key in (ord("p"), ord("P")):
                    print(
                        "Current thresholds -> "
                        f"yaw={config.yaw_threshold:.2f}, pitch={config.pitch_threshold:.2f}, "
                        f"gaze_x={config.gaze_x_threshold:.2f}, gaze_y={config.gaze_y_threshold:.2f}, "
                        f"missing_face_s={config.missing_face_seconds:.1f}, "
                        f"brief_s={config.brief_away_seconds:.1f}, distracted_s={config.distracted_seconds:.1f}"
                    )
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    finally:
        detector.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 attention classification and tuning")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without display window for a fixed number of frames",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames to process in --no-gui mode (default 30)",
    )
    args = parser.parse_args()
    return run(camera_index=args.camera, no_gui=args.no_gui, frames=args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
