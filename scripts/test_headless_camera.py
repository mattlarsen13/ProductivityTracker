"""
Headless camera test utility.

This keeps the webcam open for a fixed duration (or frame count), prints
capture stats, and can save a snapshot for quick verification without GUI.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.camera import Camera, CameraError


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless camera validation runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Maximum number of frames to capture (default 300)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds (0 disables time limit)",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        default="",
        help="Optional output image path for first captured frame",
    )
    args = parser.parse_args()

    camera = Camera(camera_index=args.camera)
    start = time.time()
    last_log = start
    count = 0
    first_frame_saved = False

    try:
        camera.open()
        print(f"Opened camera index {args.camera} in headless mode.")
        while count < args.frames:
            if args.seconds > 0 and (time.time() - start) >= args.seconds:
                break

            frame = camera.read()
            count += 1

            if args.save_frame and not first_frame_saved:
                out_path = Path(args.save_frame)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), frame)
                print(f"Saved first frame to: {out_path}")
                first_frame_saved = True

            now = time.time()
            if (now - last_log) >= 1.0:
                elapsed = max(now - start, 1e-6)
                fps = count / elapsed
                print(f"Frames={count} | Elapsed={elapsed:.1f}s | AvgFPS={fps:.1f}")
                last_log = now

        elapsed = max(time.time() - start, 1e-6)
        fps = count / elapsed
        print(f"Done. Captured {count} frames in {elapsed:.2f}s (AvgFPS={fps:.2f}).")
        return 0
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
