"""
Minimal webcam check: opens the default camera, reads frames, optionally previews.

Usage:
  python scripts/smoke_webcam.py              # preview window; press Q to quit
  python scripts/smoke_webcam.py --no-gui   # no window; reads N frames and exits
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCV webcam smoke test")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open a window; only verify capture for N frames",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames to read in --no-gui mode (default 30)",
    )
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Activate your venv and run: pip install -r requirements.txt")
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}. Try --camera 1 if you have multiple devices.")
        return 1

    if args.no_gui:
        ok = 0
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Frame {i}: read failed")
                break
            ok += 1
            if i == 0:
                h, w = frame.shape[:2]
                print(f"First frame: shape={frame.shape} (H={h}, W={w})")
        cap.release()
        if ok == args.frames:
            print(f"OK: captured {ok} frames without GUI.")
            return 0
        print(f"Partial failure: only {ok}/{args.frames} frames captured.")
        return 1

    print("Preview window open — press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame read failed; exiting.")
            break
        cv2.imshow("smoke_webcam (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
