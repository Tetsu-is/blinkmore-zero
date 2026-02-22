# blinkmore-zero

Blink detection for Linux + Polybar using MediaPipe FaceLandmarker and OpenCV.

Tracks your eye blink interval via webcam and displays it live in your Polybar status bar.

Inspired by [BlinkMore](https://github.com/oxremy/BlinkMore), a macOS app that encourages you to blink more often.

## How it works

`blink_detector.py` reads webcam frames and detects blinks using the Eye Aspect Ratio (EAR) method:

- MediaPipe FaceLandmarker extracts 478 face landmarks in VIDEO mode
- EAR is computed from 6 eye landmarks per eye; a blink is detected when EAR drops below 0.2 for at least 2 consecutive frames
- State is written to `~/.cache/blink_state.json` every 0.1s

`polybar_blink.sh` reads that JSON file via `jq` and outputs the blink interval to Polybar.

## Requirements

- Python >= 3.14
- `jq`
- A webcam

## Installation

```sh
# Download the MediaPipe face landmarker model
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Install Python dependencies (using uv)
uv sync
```

## Usage

Run the detector in the background:

```sh
uv run blink_detector.py &
```

Add the Polybar module to your Polybar config:

```ini
[module/blink]
type = custom/script
exec = /path/to/polybar_blink.sh
interval = 1
```

Then include `blink` in your bar's `modules-right` (or wherever you want it).

## State file

`~/.cache/blink_state.json` format:

```json
{
  "eyes_open": true,
  "interval": 3.42,
  "last_update": "2026-02-22T12:34:56.789",
  "timestamp": 1740000000.0
}
```

The Polybar script treats the detector as offline if the timestamp is more than 15 seconds old.

## Eye landmark indices (MediaPipe)

| Eye   | Indices                      |
|-------|------------------------------|
| Left  | 33, 160, 158, 133, 153, 144  |
| Right | 362, 385, 387, 263, 373, 380 |
