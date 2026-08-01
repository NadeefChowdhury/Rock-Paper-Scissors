"""
Rock Paper Scissors — Computer Vision Edition
================================================
Play Rock/Paper/Scissors against the computer using your webcam.
Your hand gesture is detected with MediaPipe's HandLandmarker model
and classified with OpenCV, drawing the whole game UI on top of the
live camera feed. First to 3 round wins takes the match.

Controls
--------
SPACE : start a round / play again after a match ends / skip calibration
P     : toggle Practice Mode (from idle)
C     : re-run calibration (from idle)
Q     : quit

Game flow
---------
1. CALIBRATION: hold up Rock, then Paper, then Scissors so the game can
   confirm gesture detection is working before you start playing.
2. IDLE: press SPACE to start a round, or P to enter Practice Mode.
3. COUNTDOWN: "3, 2, 1, Shoot!" with a rock/paper/scissors image flicker.
4. PROCESSING: your gesture is locked in (using a short stability check
   so a mid-transition frame can't get grabbed by accident); the
   computer's choice is shown while "Checking..." plays.
5. REVEAL: winner is announced with a color flash and the round is
   scored. First to MATCH_WIN_SCORE round wins ends the match.

First run
---------
The hand-tracking model (~10 MB) is downloaded automatically the
first time you run this script and cached locally as
`hand_landmarker.task`.
"""

import os
import random
import time
import urllib.request
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Setup: download the hand landmark model if we don't already have it
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


# --------------------------------------------------------------------------
# Gesture classification
# --------------------------------------------------------------------------
# MediaPipe hand landmark indices we need:
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_TIPS_PIPS = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]


def finger_states(landmarks, handedness_label):
    """Return a list of 5 booleans: [thumb, index, middle, ring, pinky]
    where True means the finger is extended."""
    states = []

    # Thumb: compare x position of tip vs. its IP joint. Direction flips
    # depending on whether MediaPipe reports this as the Left or Right hand
    # (note: MediaPipe's labels are mirrored relative to a selfie-view feed).
    if handedness_label == "Right":
        thumb_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    else:
        thumb_extended = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
    states.append(thumb_extended)

    # Other four fingers: extended if the tip is above (smaller y) the PIP joint.
    for tip_idx, pip_idx in FINGER_TIPS_PIPS:
        states.append(landmarks[tip_idx].y < landmarks[pip_idx].y)

    return states


def classify_gesture(landmarks, handedness_label):
    thumb, index, middle, ring, pinky = finger_states(landmarks, handedness_label)
    extended_count = sum([thumb, index, middle, ring, pinky])
    non_thumb_extended = sum([index, middle, ring, pinky])

    if non_thumb_extended == 0:
        return "Rock"
    if extended_count >= 4 and non_thumb_extended >= 4:
        return "Paper"
    if index and middle and not ring and not pinky:
        return "Scissors"
    return None  # unrecognized / hand still moving into position


def pick_best_hand(result):
    """From a (possibly two-hand) detection result, pick the hand MediaPipe
    is most confident about. Returns (landmarks, handedness_label, confidence)
    or (None, None, 0.0) if no hand was found."""
    if not result.hand_landmarks:
        return None, None, 0.0
    best_idx = max(range(len(result.handedness)), key=lambda i: result.handedness[i][0].score)
    landmarks = result.hand_landmarks[best_idx]
    handedness_label = result.handedness[best_idx][0].category_name
    confidence = result.handedness[best_idx][0].score
    return landmarks, handedness_label, confidence, best_idx


# --------------------------------------------------------------------------
# Game logic
# --------------------------------------------------------------------------
CHOICES = ["Rock", "Paper", "Scissors"]
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
MATCH_WIN_SCORE = 3  # first player to reach this many round wins takes the match

# Anti-cheat / stability check: rather than trusting whatever gesture was
# seen on the very last frame, we require the majority of gestures seen in
# a short trailing window to agree before locking in the player's move.
STABILITY_WINDOW = 0.3  # seconds of recent history to look at
STABILITY_AGREEMENT_RATIO = 0.6  # fraction of that window that must agree


def decide_winner(player, computer):
    if player == computer:
        return "Tie"
    if BEATS[player] == computer:
        return "Player"
    return "Computer"


def stable_gesture(gesture_history, now, fallback):
    """Look at recent (timestamp, gesture) history and return the gesture
    that dominated the last STABILITY_WINDOW seconds, or `fallback` if the
    hand was missing or the signal was too noisy/ambiguous to trust."""
    recent = [g for t, g in gesture_history if now - t <= STABILITY_WINDOW]
    if not recent:
        return fallback
    gesture, count = Counter(recent).most_common(1)[0]
    if count / len(recent) >= STABILITY_AGREEMENT_RATIO:
        return gesture
    return fallback


# --------------------------------------------------------------------------
# UI drawing helpers
# --------------------------------------------------------------------------
def draw_text(img, text, pos, scale=1.0, color=(255, 255, 255), thickness=2, center=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if center:
        x -= w // 2
    # subtle drop-shadow for readability over the camera feed
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_animated_text(img, text, pos, progress, base_scale, color, thickness):
    """Draws text with a quick pop-in / pop-out animation.
    `progress` is 0..1 through however long the text should be shown for."""
    if progress < 0.2:
        t = progress / 0.2
        scale = base_scale * (0.4 + 0.6 * t)
        alpha = t
    elif progress > 0.8:
        t = (progress - 0.8) / 0.2
        scale = base_scale * (1.0 + 0.25 * t)
        alpha = 1.0 - t
    else:
        scale = base_scale
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))

    overlay = img.copy()
    draw_text(overlay, text, pos, scale, color, thickness, center=True)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_scoreboard(img, player_score, computer_score):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 70), (30, 30, 30), -1)
    draw_text(img, f"You: {player_score}", (20, 45), 1.0, (0, 255, 120))
    draw_text(img, f"Computer: {computer_score}", (w - 250, 45), 1.0, (0, 140, 255))


def draw_hand_status(img, hand_detected, confidence):
    """Small always-on indicator of whether/how confidently a hand is seen."""
    x, y = 20, img.shape[0] - 15
    if hand_detected:
        draw_text(img, f"Hand: {confidence * 100:.0f}%", (x, y), 0.6, (0, 255, 120), 1)
    else:
        draw_text(img, "Hand: not detected", (x, y), 0.6, (0, 0, 255), 1)


def draw_progress_bar(img, x, y, width, height, fraction, color):
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(width * fraction), y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)


def apply_flash(img, color, alpha):
    if alpha <= 0:
        return
    tint = np.full_like(img, color, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, img, 1 - alpha, 0, dst=img)


def overlay_image(background, overlay, x, y):
    """Draw 'overlay' onto 'background' with its top-left corner at (x, y).
    Supports both regular BGR images and BGRA images with transparency."""
    if overlay is None:
        return

    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh:
        return
    w = min(w, bw - x)
    h = min(h, bh - y)
    overlay = overlay[:h, :w]

    if overlay.shape[2] == 4:  # has an alpha channel -> blend
        alpha = overlay[:, :, 3:4] / 255.0
        region = background[y:y + h, x:x + w].astype(float)
        blended = overlay[:, :, :3].astype(float) * alpha + region * (1 - alpha)
        background[y:y + h, x:x + w] = blended.astype("uint8")
    else:  # plain BGR -> just paste it
        background[y:y + h, x:x + w] = overlay[:, :, :3]


# --------------------------------------------------------------------------
# Computer's move images (shown when its choice is revealed)
# --------------------------------------------------------------------------
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_IMAGE_PATHS = {
    "Rock": os.path.join(ASSETS_DIR, "rock.jpg"),
    "Paper": os.path.join(ASSETS_DIR, "paper.jpg"),
    "Scissors": os.path.join(ASSETS_DIR, "scissors.png"),
}
GESTURE_IMAGE_SIZE = 220  # displayed as a GESTURE_IMAGE_SIZE x GESTURE_IMAGE_SIZE square


def load_gesture_images():
    images = {}
    for name, path in GESTURE_IMAGE_PATHS.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not load '{path}'. The {name} image won't be shown.")
            images[name] = None
            continue
        images[name] = cv2.resize(img, (GESTURE_IMAGE_SIZE, GESTURE_IMAGE_SIZE))
    return images


# --------------------------------------------------------------------------
# Main game loop
# --------------------------------------------------------------------------
STATE_CALIBRATION = "calibration"
STATE_IDLE = "idle"
STATE_PRACTICE = "practice"
STATE_COUNTDOWN = "countdown"
STATE_PROCESSING = "processing"  # brief pause after "Shoot!" while the computer "decides"
STATE_REVEAL = "reveal"
STATE_MATCH_OVER = "match_over"  # someone reached MATCH_WIN_SCORE

CALIBRATION_SEQUENCE = ["Rock", "Paper", "Scissors"]
CALIBRATION_HOLD_DURATION = 1.0  # seconds a gesture must be held to pass

COUNTDOWN_STEPS = ["3", "2", "1", "Shoot!"]
STEP_DURATION = 0.7  # seconds per countdown step
CYCLE_INTERVAL = 0.1  # seconds between image swaps in the rock/paper/scissors flicker
PROCESSING_DURATION = 1.0  # seconds spent "checking" before the reveal
REVEAL_DURATION = 2.0  # seconds to show the round result

FLASH_DURATION = 0.35  # seconds the win/loss/tie color flash lasts
FLASH_MAX_ALPHA = 0.35
FLASH_COLORS = {  # BGR
    "You win!": (0, 200, 0),
    "Computer wins!": (0, 0, 220),
    "Tie!": (0, 210, 210),
}

# Requested webcam capture resolution. Most webcams support 1280x720;
# if yours doesn't, OpenCV will fall back to its closest supported size.
CAM_WIDTH = 1280
CAM_HEIGHT = 720
WINDOW_NAME = "Rock Paper Scissors - CV Edition"


def main():
    ensure_model()
    gesture_images = load_gesture_images()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,  # detect up to two hands; we'll pick the most confident one
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check your camera permissions/index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAM_WIDTH, CAM_HEIGHT)

    player_score = 0
    computer_score = 0

    state = STATE_CALIBRATION
    state_start = time.time()
    last_seen_gesture = None
    gesture_history = deque(maxlen=90)  # (timestamp, gesture) pairs, ~3s of history
    player_choice = None
    computer_choice = None
    result_text = ""
    match_winner = None

    calibration_index = 0
    calibration_hold_start = None

    frame_timestamp_ms = 0

    print("Calibration: hold up Rock, Paper, then Scissors. Press SPACE to skip.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # assume ~30fps for the VIDEO-mode timestamp
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        now = time.time()

        # --- hand detection: pick the most confident of up to two hands ---
        detected_gesture = None
        hand_detected = False
        hand_confidence = 0.0
        picked = pick_best_hand(result)
        if picked[0] is not None:
            landmarks, handedness_label, hand_confidence, best_idx = picked
            hand_detected = True
            detected_gesture = classify_gesture(landmarks, handedness_label)

            h, w = frame.shape[:2]
            for i, lm_list in enumerate(result.hand_landmarks):
                is_chosen = (i == best_idx)
                color = (0, 255, 255) if is_chosen else (130, 130, 130)
                radius = 4 if is_chosen else 3
                for lm in lm_list:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, color, -1)

        if detected_gesture:
            last_seen_gesture = detected_gesture
            gesture_history.append((now, detected_gesture))

        elapsed = now - state_start

        # ----------------------------------------------------------------
        if state == STATE_CALIBRATION:
            target = CALIBRATION_SEQUENCE[calibration_index]
            draw_text(frame, "CALIBRATION", (frame.shape[1] // 2, 60), 1.1, (0, 220, 255), 2, center=True)
            draw_text(frame, f"Show me a {target}", (frame.shape[1] // 2, 110), 1.0, (255, 255, 255), 2, center=True)
            draw_text(frame, "(SPACE to skip)", (frame.shape[1] // 2, 145), 0.7, (0, 250, 250), 1, center=True)

            if detected_gesture == target:
                if calibration_hold_start is None:
                    calibration_hold_start = now
                hold_progress = (now - calibration_hold_start) / CALIBRATION_HOLD_DURATION
                draw_progress_bar(frame, frame.shape[1] // 2 - 150, 170, 300, 24, hold_progress, (0, 255, 120))
                if hold_progress >= 1.0:
                    calibration_index += 1
                    calibration_hold_start = None
                    if calibration_index >= len(CALIBRATION_SEQUENCE):
                        state = STATE_IDLE
                        state_start = now
            else:
                calibration_hold_start = None
                draw_progress_bar(frame, frame.shape[1] // 2 - 150, 170, 300, 24, 0.0, (0, 255, 120))

        # ----------------------------------------------------------------
        elif state == STATE_IDLE:
            draw_text(frame, "Press SPACE to play", (frame.shape[1] // 2, 100), 1.1, (255, 255, 255), 2, center=True)
            draw_text(frame, "P: Practice mode   C: Recalibrate", (frame.shape[1] // 2, 135), 0.6, (0, 250, 250), 1, center=True)
            if detected_gesture:
                draw_text(frame, f"I see: {detected_gesture}", (frame.shape[1] // 2, 170), 0.9, (0, 255, 255), 2, center=True)
            draw_hand_status(frame, hand_detected, hand_confidence)

        # ----------------------------------------------------------------
        elif state == STATE_PRACTICE:
            draw_text(frame, "PRACTICE MODE", (frame.shape[1] // 2, 100), 1.1, (0, 220, 255), 2, center=True)
            draw_text(frame, "Press P to exit", (frame.shape[1] // 2, 135), 0.7, (200, 200, 200), 1, center=True)
            if detected_gesture:
                draw_text(frame, detected_gesture, (frame.shape[1] // 2, 200), 1.6, (0, 255, 120), 3, center=True)
            else:
                draw_text(frame, "No gesture recognized", (frame.shape[1] // 2, 200), 0.9, (0, 0, 255), 2, center=True)
            draw_hand_status(frame, hand_detected, hand_confidence)

        # ----------------------------------------------------------------
        elif state == STATE_COUNTDOWN:
            # rapid rock/paper/scissors flicker to build suspense
            flicker_choice = CHOICES[int(elapsed / CYCLE_INTERVAL) % len(CHOICES)]
            flicker_img = gesture_images.get(flicker_choice)
            img_x = frame.shape[1] - GESTURE_IMAGE_SIZE - 20
            img_y = 90
            overlay_image(frame, flicker_img, img_x, img_y)

            if not hand_detected:
                draw_text(frame, "Show your hand!", (frame.shape[1] // 2, 220), 0.9, (0, 0, 255), 2, center=True)

            step = int(elapsed // STEP_DURATION)
            if step >= len(COUNTDOWN_STEPS):
                # lock in the player's gesture right as we hit "Shoot!", using
                # the stability check so a mid-transition frame can't sneak in.
                player_choice = stable_gesture(gesture_history, now, last_seen_gesture or "Rock")
                computer_choice = random.choice(CHOICES)
                state = STATE_PROCESSING
                state_start = now
            else:
                step_progress = (elapsed % STEP_DURATION) / STEP_DURATION
                draw_animated_text(frame, COUNTDOWN_STEPS[step], (frame.shape[1] // 2, 170), step_progress, 2.2, (0, 200, 255), 4)

            draw_hand_status(frame, hand_detected, hand_confidence)

        # ----------------------------------------------------------------
        elif state == STATE_PROCESSING:
            # computer's choice is already locked in — show it plainly while "checking"
            img_x = frame.shape[1] - GESTURE_IMAGE_SIZE - 20
            img_y = 90
            overlay_image(frame, gesture_images.get(computer_choice), img_x, img_y)

            draw_text(frame, f"You: {player_choice}", (frame.shape[1] // 2, 100), 1.0, (0, 255, 120), 2, center=True)
            draw_text(frame, f"Computer: {computer_choice}", (frame.shape[1] // 2, 140), 1.0, (0, 140, 255), 2, center=True)
            draw_text(frame, "Checking...", (frame.shape[1] // 2, 190), 1.3, (0, 200, 255), 3, center=True)
            if elapsed > PROCESSING_DURATION:
                winner = decide_winner(player_choice, computer_choice)
                if winner == "Player":
                    player_score += 1
                    result_text = "You win!"
                elif winner == "Computer":
                    computer_score += 1
                    result_text = "Computer wins!"
                else:
                    result_text = "Tie!"
                state = STATE_REVEAL
                state_start = now

        # ----------------------------------------------------------------
        elif state == STATE_REVEAL:
            flash_alpha = FLASH_MAX_ALPHA * max(0.0, 1.0 - elapsed / FLASH_DURATION)
            apply_flash(frame, FLASH_COLORS.get(result_text, (255, 255, 255)), flash_alpha)

            img = gesture_images.get(computer_choice)
            img_x = frame.shape[1] - GESTURE_IMAGE_SIZE - 20
            img_y = 90
            overlay_image(frame, img, img_x, img_y)

            draw_text(frame, f"You: {player_choice}", (frame.shape[1] // 2, 100), 1.0, (0, 255, 120), 2, center=True)
            draw_text(frame, f"Computer: {computer_choice}", (frame.shape[1] // 2, 140), 1.0, (0, 140, 255), 2, center=True)
            draw_text(frame, result_text, (frame.shape[1] // 2, 190), 1.3, (255, 255, 255), 3, center=True)

            if elapsed > REVEAL_DURATION:
                if player_score >= MATCH_WIN_SCORE or computer_score >= MATCH_WIN_SCORE:
                    match_winner = "You" if player_score >= MATCH_WIN_SCORE else "Computer"
                    state = STATE_MATCH_OVER
                else:
                    state = STATE_IDLE
                state_start = now
                last_seen_gesture = None
                gesture_history.clear()

        # ----------------------------------------------------------------
        elif state == STATE_MATCH_OVER:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

            win_color = (0, 220, 255) if match_winner == "You" else (0, 140, 255)
            draw_text(frame, "MATCH OVER", (frame.shape[1] // 2, frame.shape[0] // 2 - 90), 1.2, (255, 255, 255), 2, center=True)
            draw_text(frame, f"{match_winner} won the match!", (frame.shape[1] // 2, frame.shape[0] // 2 - 40), 1.5, win_color, 3, center=True)
            draw_text(frame, f"Final score  —  You: {player_score}   Computer: {computer_score}",
                      (frame.shape[1] // 2, frame.shape[0] // 2 + 10), 0.9, (255, 255, 255), 2, center=True)
            draw_text(frame, "Press SPACE to play again", (frame.shape[1] // 2, frame.shape[0] // 2 + 60), 0.8, (200, 200, 200), 2, center=True)
            draw_text(frame, "Press Q to quit", (frame.shape[1] // 2, frame.shape[0] // 2 + 95), 0.8, (200, 200, 200), 2, center=True)

        # ----------------------------------------------------------------
        if state not in (STATE_CALIBRATION, STATE_MATCH_OVER):
            draw_scoreboard(frame, player_score, computer_score)
        draw_text(frame, "Q: quit", (frame.shape[1] - 130, frame.shape[0] - 15), 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            if state == STATE_CALIBRATION:
                state = STATE_IDLE
                state_start = time.time()
            elif state == STATE_IDLE:
                state = STATE_COUNTDOWN
                state_start = time.time()
                last_seen_gesture = None
                gesture_history.clear()
            elif state == STATE_MATCH_OVER:
                player_score = 0
                computer_score = 0
                match_winner = None
                state = STATE_IDLE
                state_start = time.time()

        elif key == ord("p") and state in (STATE_IDLE, STATE_PRACTICE):
            state = STATE_PRACTICE if state == STATE_IDLE else STATE_IDLE
            state_start = time.time()

        elif key == ord("c") and state == STATE_IDLE:
            state = STATE_CALIBRATION
            state_start = time.time()
            calibration_index = 0
            calibration_hold_start = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()