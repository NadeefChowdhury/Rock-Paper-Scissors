# Rock-Paper-Scissors
This is a game where you can play Rock-Paper-Scissors against the computer using hand gestures detected by your webcam, powered by OpenCV and MediaPipe.

## Setup

```bash
pip install -r requirements.txt
python game.py
```

The hand-tracking model (10 MB) downloads automatically the first time you run the script and is cached as `hand_landmarker.task` next to the script, so later runs work offline.

## How to play

1. A window opens showing your webcam feed with your hand landmarks
   overlaid (yellow dots) whenever a hand is detected.
2. Hold your hand up so the camera can see it, then calibrate so that the program can detect your hand signs properly.
3. In the next step, there are two options: P for Practice and C for Calibrating again. You can also start playing.
4. A "3, 2, 1, Shoot!" countdown plays. Make your gesture right as
   "Shoot!" appears:
   - **Fist** → Rock
   - **Open palm** → Paper
   - **V sign** → Scissors
5. Your gesture and the computer's random pick are revealed, the round winner is decided, and the scoreboard updates.
6. Press **SPACE** to play again, or **Q** to quit at any time.
7. The first one to reach 3 points wins the match.

## Tips for reliable detection

- Keep your hand centered in frame, roughly arm's length from the camera.
- Hold the gesture steady for a beat right as "Shoot!" appears; the game uses whichever gesture was most recently recognized at that instant.

## How the gesture detection works

MediaPipe's `HandLandmarker` returns 21 3D landmark points per hand (fingertips, knuckles, wrist, etc.). For each frame we check, per finger, whether the tip is "extended" (roughly straight) or "curled":

- For the four fingers (index/middle/ring/pinky): the tip is considered extended if it's above its own middle knuckle (PIP joint) in the image.
- For the thumb: extension is checked sideways (x-axis) relative to its IP joint, and the direction is flipped depending on whether MediaPipe reports the hand as Left or Right (since the camera feed is mirrored for a natural selfie view).

From there:
- **0 fingers extended** (besides possibly the thumb) → **Rock**
- **Index + middle extended, ring + pinky curled** → **Scissors**
- **4-5 fingers extended** → **Paper**

Feel free to tweak the thresholds in `classify_gesture()` in `game.py` if detection feels off for your hand/camera setup.

## Troubleshooting

- **Camera doesn't open**: try changing `cv2.VideoCapture(0)` to `1` or `2` in `main()` if you have multiple cameras.
- **Model download fails**: your network may block `storage.googleapis.com`. You can manually download the model from MediaPipe's model page and place it next to `rps_game.py` as `hand_landmarker.task`.
- **Gestures misclassified**: make sure your whole hand (including wrist) is visible and well-lit; avoid busy/cluttered backgrounds.
