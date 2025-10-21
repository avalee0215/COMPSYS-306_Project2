import os, joblib, numpy as np, cv2, random
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import albumentations as A

# ------------------------------
# Configuration
# ------------------------------
DATASET_PATH = 'traffic_signs_dataset_v5'
MODEL_SAVE_PATH = 'traffic_sign_model_nohog.joblib'
IMAGE_SIZE = (48, 48)   # smaller than 64x64 helps classical models
AUG_PER_IMAGE = 2       # how many augmented copies per original
USE_MODEL = 'svm'       # 'svm' or 'mlp'
RANDOM_STATE = 42

# ------------------------------
# Augmentation (safe-ish for tabletop signs/lights)
# ------------------------------
augment = A.Compose([
    A.Affine(scale=(0.9, 1.1), rotate=(-12, 12), translate_percent=(-0.06, 0.06), p=0.9),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
    A.RandomShadow(p=0.15),
    A.Perspective(scale=(0.02, 0.05), p=0.2),
])

# ------------------------------
# Feature Extraction (no HOG)
#   - normalized pixels
#   - HSV mean/std per channel
# ------------------------------
def img_to_features(img_bgr):
    img = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    # normalize pixels (0..1)
    pix = (img.astype(np.float32) / 255.0).reshape(-1)
    # color stats (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    means = hsv.mean(axis=(0,1))
    stds  = hsv.std(axis=(0,1)) + 1e-6
    color_stats = np.concatenate([means, stds]).astype(np.float32)
    return np.concatenate([pix, color_stats])

def load_dataset_with_aug(dataset_path, aug_times=AUG_PER_IMAGE):
    X, y, label_names = [], [], []
    label_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    for label_idx, label_name in enumerate(label_names):
        class_dir = os.path.join(dataset_path, label_name)
        for file in os.listdir(class_dir):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            img = cv2.imread(os.path.join(class_dir, file))
            if img is None: 
                continue

            # original
            X.append(img_to_features(img))
            y.append(label_idx)

            # augmented copies
            h, w = img.shape[:2]
            for _ in range(aug_times):
                aug = augment(image=img)['image']
                X.append(img_to_features(aug))
                y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), label_names

# ------------------------------
# Cleanup helpers (optional)
# ------------------------------
def clean_dataset(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == '.DS_Store':
                try:
                    os.remove(os.path.join(root, f))
                except: pass
        # don't remove checkpoints here to avoid permission issues

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("Cleaning dataset directory...")
    clean_dataset(DATASET_PATH)

    print("Loading + augmenting dataset...")
    X, y, label_names = load_dataset_with_aug(DATASET_PATH, AUG_PER_IMAGE)
    print(f"Loaded {len(y)} samples (with augmentation), {len(label_names)} classes")

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --------------------------
    # Pipelines + hyperparameter search
    # --------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if USE_MODEL == 'svm':
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', svm.SVC(kernel='rbf', probability=True, class_weight='balanced'))
        ])
        param_grid = {
            'clf__C':     [0.5, 1, 2, 4, 8, 16],
            'clf__gamma': ['scale', 0.01, 0.005, 0.001, 0.0005]
        }
        search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                max_iter=60,        # keep small; early stopping via tol
                random_state=RANDOM_STATE,
                verbose=False
            ))
        ])
        param_grid = {
            'clf__hidden_layer_sizes': [(256,128), (256,), (384,192), (256,128,64)],
            'clf__alpha': [1e-5, 1e-4, 1e-3],
            'clf__learning_rate_init': [1e-3, 5e-4, 1e-4]
        }
        search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)

    print("Tuning hyperparameters...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    best = search.best_estimator_

    # Evaluation
    y_pred = best.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save (keep same format you already use)
    joblib.dump({'model': best, 'labels': label_names, 'image_size': IMAGE_SIZE}, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
