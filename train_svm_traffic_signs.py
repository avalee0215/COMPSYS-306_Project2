# train_svm_traffic_signs_rgb_hist.py

import os
import joblib
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ------------------------------
# Configuration
# ------------------------------
DATASET_PATH = 'dataset'
MODEL_SAVE_PATH = 'svm_traffic_sign_rgb_hist.joblib'
IMAGE_SIZE = (64, 64)  # resize for feature extraction
BINS = 16  # number of bins per color channel

# ------------------------------
# Load dataset and extract RGB histogram features
# ------------------------------
def load_dataset(dataset_path):
    X = []
    y = []
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
                img = cv2.resize(img, IMAGE_SIZE)
                
                # Extract RGB histograms
                channels = cv2.split(img)
                features = []
                for ch in channels:
                    hist = cv2.calcHist([ch], [0], None, [BINS], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    features.extend(hist)
                features = np.array(features)
                
                X.append(features)
                y.append(label_idx)
    
    return np.array(X), np.array(y), label_names

# ------------------------------
# Main training process
# ------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    X, y, label_names = load_dataset(DATASET_PATH)
    print(f"Dataset loaded: {X.shape[0]} samples, {len(label_names)} classes")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #pca = PCA(n_components=100)
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)
    
    print("Training SVM model...")
    clf = svm.SVC(kernel='rbf', probability=True, gamma='scale')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model and label mapping
    joblib.dump({'model': clf, 'labels': label_names}, MODEL_SAVE_PATH)
    print(f"\nModel saved as {MODEL_SAVE_PATH}")
