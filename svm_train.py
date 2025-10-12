"""
svm_train.py

Train SVM classifier on flattened images using scikit-learn.
Because raw image dimensionality is large, PCA is applied for speed.
Performs GridSearchCV on C and gamma (for rbf), and saves model and metrics.
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configuration Arguments to run program
DATA_NPZ = "./processed/dataset.npz"   # Path to dataset.npz
OUT_DIR  = "./results/svm"             # Output folder
PCA_COMPONENTS = 100                   # Number of PCA components
CV_FOLDS = 3                           # Cross-validation folds
N_JOBS = -1                            # Parallel jobs for GridSearchCV

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset from .npz
    data = np.load(DATA_NPZ, allow_pickle=True)
    X_train = data['X_train']  # (N,H,W,3)
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    classes = list(data['classes'])

    # Flatten images
    def prepare(X):
        return X.reshape(X.shape[0], -1).astype(np.float32)
    X_train_f = prepare(X_train)
    X_val_f = prepare(X_val)
    X_test_f = prepare(X_test)

    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    scaler.fit(X_train_f)
    X_train_s = scaler.transform(X_train_f)
    X_val_s = scaler.transform(X_val_f)
    X_test_s = scaler.transform(X_test_f)

    # Apply PCA + SVM
    steps = []
    if PCA_COMPONENTS and PCA_COMPONENTS > 0:
        steps.append(('pca', PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=42)))
    steps.append(('svc', SVC(probability=False)))
    pipe = Pipeline(steps)

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'svc__C': [1.0, 10.0],           # regularization strength
        'svc__kernel': ['rbf'],          # radial basis function kernel
        'svc__gamma': ['scale', 0.01]    # kernel coefficient
    }
    grid = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, verbose=2, n_jobs=N_JOBS)
    print("Starting GridSearchCV (may take a while)...")
    grid.fit(X_train_s, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # Evaluate on validation set
    y_val_pred = best.predict(X_val_s)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
    print("Validation acc:", val_acc, "prec:", val_prec, "rec", val_rec, "f1:", val_f1)

    # Evaluate on test set
    y_test_pred = best.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)
    print("Test acc:", test_acc, "prec:", test_prec, "rec:", test_rec, "f1:", test_f1)

    # Save model, scaler, classes, and PCA info
    joblib.dump({
        'model': best,
        'scaler': scaler,
        'classes': np.array(classes),
        'pca_components': PCA_COMPONENTS
    }, os.path.join(OUT_DIR, 'best_svm.pkl'))

    # Save test metrics
    np.savez_compressed(os.path.join(OUT_DIR, 'svm_metrics.npz'),
                        test_acc=test_acc, test_prec=test_prec, test_rec=test_rec, test_f1=test_f1, confusion_matrix=cm, classes=np.array(classes))
    print("Saved SVM and metrics to", OUT_DIR)

if __name__ == '__main__':
    main()