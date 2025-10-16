import cv2
import joblib
import time
import numpy as np
from jetbot import Camera
import lfrobot

# ------------------------------
# Config
# ------------------------------
MODEL_PATH = 'svm_traffic_sign_rgb_hist.joblib'  # updated to your new model
IMAGE_SIZE = (64, 64)
BINS = 16

# Load model
data = joblib.load(MODEL_PATH)
model = data['model']
labels = data['labels']
print("Loaded model with labels:", labels)

# Initialize camera and robot
camera = Camera.instance(width=224, height=224)
lfrobot.lfInit()
lfrobot.lfSpeed(0.15)
lfrobot.lfTurnSpeed(0.25)
lfrobot.lfStart()

# Helper: extract RGB histogram features
def extract_rgb_hist_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
    channels = cv2.split(img)
    features = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

# Helper: preprocess and classify frame
def classify_frame(frame):
    features = extract_rgb_hist_features(frame)
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0].max()
    return labels[pred], prob

# ------------------------------
# Main loop
# ------------------------------
try:
    while True:
        frame = camera.value
        label, confidence = classify_frame(frame)
        print(f"Detected: {label} ({confidence:.2f})")

        # Behavior logic
        if confidence < 0.6:
            continue  # ignore uncertain predictions

        if label == 'stop':
            print("STOP detected — stopping for 5s")
            lfrobot.lfStop()
            time.sleep(5)
            lfrobot.lfStart()
            lfrobot.lfSpeed(0.15)

        elif label == 'speed':
            print("SPEED sign — slowing down")
            lfrobot.lfSpeed(0.10)

        elif label == 'green':
            print("GREEN — go")
            lfrobot.lfStart()
            lfrobot.lfSpeed(0.15)

        elif label == 'yellow':
            print("YELLOW — stop")
            lfrobot.lfStop()
            time.sleep(2)

        elif label == 'sheep':
            print("SHEEP — stop indefinitely")
            lfrobot.lfStop()
            while True:
                frame2 = camera.value
                l2, c2 = classify_frame(frame2)
                if l2 != 'sheep' or c2 < 0.6:
                    print("Sheep gone — resume")
                    lfrobot.lfStart()
                    break
                time.sleep(0.5)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopping robot")
    lfrobot.lfDeinit()
    camera.stop()
