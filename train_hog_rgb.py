import os
import joblib
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# Configuration
DATASET_PATH = 'traffic_signs_dataset_v21'
MODEL_SAVE_PATH = 'svm_traffic_sign_rgb_hog.joblib'
IMAGE_SIZE = (64, 64)
# Histogram bins
BINS = 16  

# Feature Extraction Function
def extract_features(img):
    img = cv2.resize(img, IMAGE_SIZE)

    # RGB histogram
    channels = cv2.split(img)
    hist_features = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    # HOG features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)

    # Combine
    return np.concatenate([hist_features, hog_features])

# Load dataset
def load_dataset(dataset_path):
    X, y = [], []
    label_names = sorted(os.listdir(dataset_path))
    print("Detected classes:", label_names)

    for label_idx, label_name in enumerate(label_names):
        label_dir = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_dir):
            continue

        for file in os.listdir(label_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_dir, file)
                img = cv2.imread(img_path)
                features = extract_features(img)

                X.append(features)
                y.append(label_idx)

    return np.array(X), np.array(y), label_names

# Cleanup unwanted files/folders
def clean_dataset(path):
    for root, dirs, files in os.walk(path):
        # Delete unwanted files
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        # Delete unwanted directories
        for dir_name in dirs:
            if dir_name == '.ipynb_checkpoints':
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except OSError:
                    print(f"Could not delete folder (not empty): {dir_path}")

# Main
if __name__ == "__main__":
    print("Cleaning dataset directory...")
    clean_dataset(DATASET_PATH)

    print("Loading dataset...")
    X, y, label_names = load_dataset(DATASET_PATH)
    print(f"Loaded {X.shape[0]} samples, {len(label_names)} classes")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    print("Training SVM...")
    clf = svm.SVC(kernel='rbf', probability=True, gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump({'model': clf, 'labels': label_names}, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
