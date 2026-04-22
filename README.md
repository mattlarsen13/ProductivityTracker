# ProductivityTracker
Using OpenCV to keep you productive.

## Step 1: Python Environment + Webcam Smoke Test

This milestone validates local setup only:
- Python virtual environment is created.
- Dependencies install successfully.
- Webcam can be opened and frames can be captured.

### Prerequisites (Windows)
- Python 3.11+ installed and available as `python`
- Webcam connected and not locked by another app

### Setup Commands (PowerShell)
From the project root:

```powershell
# Prefer py launcher on Windows
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Run Smoke Test

Headless check (recommended first):

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_webcam.py --no-gui
```

Preview window check:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_webcam.py
```

If your default camera is not index 0:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_webcam.py --camera 1
```

## Troubleshooting
- `Could not open camera index 0`: close Zoom/Teams/Discord/OBS and retry.
- OpenCV import error: ensure you are using `.venv` Python and reinstall requirements.
- Black screen or frame failures: try another index (`--camera 1`, `--camera 2`) and check camera privacy settings in Windows.

## Step 2: Baseline Face + Eye Tracking

This milestone adds:
- Face detection (single face) using OpenCV Haar cascades.
- Eye detection and eye openness proxy (EAR-like ratio) for both eyes.
- Pupil centroid offset as a lightweight gaze-direction proxy.
- Head yaw/pitch proxy from face position in frame.

### Install New Dependency

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Run Baseline Tracker

```powershell
.\.venv\Scripts\python.exe .\src\main.py
```

Optional camera index:

```powershell
.\.venv\Scripts\python.exe .\src\main.py --camera 1
```

Headless quick check:

```powershell
.\.venv\Scripts\python.exe .\src\main.py --no-gui --frames 30
```

Dedicated headless camera tester (no tracker overlays):

```powershell
.\.venv\Scripts\python.exe .\scripts\test_headless_camera.py --seconds 10 --save-frame .\artifacts\headless_frame.jpg
```

### What You Should See
- A webcam preview with face and eye detection boxes.
- Live debug text for:
  - `Left eye EAR`, `Right eye EAR`
  - `gaze(x,y)` proxy per eye
  - `head yaw/pitch` proxy
- Press `Q` to close.

## Step 3: Attention Classification + Live Tuning

This milestone adds real-time classification states:
- `focused`
- `brief_away`
- `distracted`

Classification is based on yaw/pitch, gaze proxy, and missing-face duration.

### Run Step 3

```powershell
.\.venv\Scripts\python.exe .\src\main.py
```

### Live Tuning Controls (while camera window is open)
- `1` select `yaw_threshold`
- `2` select `pitch_threshold`
- `3` select `gaze_x_threshold`
- `4` select `gaze_y_threshold`
- `5` select `missing_face_seconds`
- `6` select `brief_away_seconds`
- `7` select `distracted_seconds`
- `Up Arrow` increase selected threshold
- `Down Arrow` decrease selected threshold
- `R` reset thresholds to defaults
- `P` print current thresholds to terminal
- `Q` quit

As you turn your head or look away, the overlay updates the current state and reason in real time.

### Optional Headless Step 3 Check

```powershell
.\.venv\Scripts\python.exe .\src\main.py --no-gui --frames 60
```
