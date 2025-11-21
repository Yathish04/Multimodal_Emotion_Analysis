# realtime_emotion.py
# Works on Windows + VS Code + DeepFace + OpenCV
# Colored boxes per emotion + colored text background

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF logs

import cv2
import numpy as np
from collections import deque
from deepface import DeepFace

# -------- SETTINGS (tweak if you want) --------
DETECTOR = "opencv"      # fastest; keep this unless you need stronger detection
FRAME_SKIP = 2           # analyze 1 of every N frames (increase to 3 if slow)
SMOOTH_WINDOW = 5        # majority vote over last N labels
UNCERTAIN_THRESH = 0.50  # if max prob < this → "uncertain"
PREVIEW_WIDTH = 960      # set 640 for extra speed
# ---------------------------------------------

# BGR colors for each emotion
EMOTION_COLORS = {
    "angry":     (0,   0, 255),   # red
    "disgust":   (147,20, 255),   # purple
    "fear":      (128, 0, 128),   # violet (similar)
    "happy":     (0, 255,   0),   # green
    "sad":       (255, 0,   0),   # blue
    "surprise":  (0, 255, 255),   # yellow
    "neutral":   (255,255, 255),  # white
    "uncertain": (128,128, 128),  # gray
}

label_history = deque(maxlen=SMOOTH_WINDOW)

def top_label(scores: dict):
    """Return (label, prob) from DeepFace 'emotion' dict (or ('unknown',0))."""
    if not scores:
        return "unknown", 0.0
    key = max(scores, key=scores.get)
    return key, float(scores[key])

def draw_box_and_label(img, region, text, color):
    """
    Draws a rectangle around the face and a filled background behind the text for readability.
    region: dict with keys x,y,w,h (may be 0 if backend didn't return a box)
    """
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    w = int(region.get("w", 0))
    h = int(region.get("h", 0))

    # Choose where to place the label text
    if w > 0 and h > 0:
        # draw face box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text_org = (x, max(20, y - 8))
    else:
        # fallback (no region): put at top-left
        text_org = (10, 30)

    # Draw filled background behind text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0, y0 = text_org
    # background rectangle coords
    bg_tl = (x0 - 4, y0 - th - 6)
    bg_br = (x0 + tw + 4, y0 + 6)
    # clamp to image bounds
    bg_tl = (max(0, bg_tl[0]), max(0, bg_tl[1]))
    bg_br = (min(img.shape[1]-1, bg_br[0]), min(img.shape[0]-1, bg_br[1]))

    # filled rectangle (slightly darker of color)
    bg_color = tuple(int(c * 0.5) for c in color)
    cv2.rectangle(img, bg_tl, bg_br, bg_color, cv2.FILLED)

    # text (white or contrasting)
    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
    cv2.putText(img, text, text_org, font, font_scale, text_color, thickness, cv2.LINE_AA)

def main():
    # try Windows-friendly camera backend first
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Close Zoom/Meet/Teams and try again.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    frame_idx = 0
    last_detections = []

    print("✅ Webcam running — press 'q' (or ESC) to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Camera read failed.")
            break

        frame_idx += 1
        display = frame.copy()

        # Use last detections on skipped frames
        detections = last_detections
        if frame_idx % FRAME_SKIP == 0:
            try:
                result = DeepFace.analyze(
                    img_path=frame,                # pass numpy array directly
                    actions=["emotion"],
                    detector_backend=DETECTOR,
                    enforce_detection=False
                )
                if isinstance(result, dict):
                    result = [result]
                detections = result
                last_detections = detections
            except Exception:
                detections = []
                last_detections = []

        if detections:
            # sort by area → define "main face"
            def area(d):
                r = d.get("region", {}) or {}
                return int(r.get("w", 0)) * int(r.get("h", 0))
            detections = sorted(detections, key=area, reverse=True)

            # smoothing label for main face
            main = detections[0]
            main_scores = main.get("emotion", {}) or {}
            dom, prob = top_label(main_scores)
            emotion_for_smooth = dom if prob >= UNCERTAIN_THRESH else "uncertain"
            label_history.append(emotion_for_smooth)
            vals, counts = np.unique(list(label_history), return_counts=True)
            smoothed = str(vals[np.argmax(counts)])

            # draw each face with its emotion color
            for det in detections:
                reg = det.get("region", {}) or {}
                scores = det.get("emotion", {}) or {}
                dlab, dprob = top_label(scores)
                emotion = dlab if dprob >= UNCERTAIN_THRESH else "uncertain"
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                text = f"{emotion} ({dprob:.2f})"
                draw_box_and_label(display, reg, text, color)

            # big smoothed label at bottom-left (white box)
            info = f"Main: {smoothed}"
            (tw, th), base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (8, display.shape[0] - th - 20), (12 + tw, display.shape[0] - 8), (0, 0, 0), cv2.FILLED)
            cv2.putText(display, info, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Real-time Emotion Detection (q / ESC to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()