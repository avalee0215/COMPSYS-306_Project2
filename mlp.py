# Import necessary libraries
import os
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skimage.io import imread
import cv2

# Define the path to your dataset
path = r''# Update this path to your dataset location
joblib_file = 'mlp_trained_model.joblib'  # Name of the trained model file
scaler_file = 'scaler.joblib'  # Name of the scaler file

# Categories (Ensure this matches your dataset folders)
CATEGORIES = [
    "dataset_55",
    "dataset_greenlight",
    "dataset_line",
    "dataset_redlight",
    "dataset_sheep",
    "dataset_stop"
]

# Function to load and preprocess the images
def load_data(path, Categories, ran_select=1000):
    flat_data_arr = []
    target_arr = []

    for i, category in enumerate(Categories):
        print(f"Loading category: {category}")
        path_i = os.path.join(path, category)
        img_files = os.listdir(path_i)

        # Randomly select images
        random_imgs = np.random.choice(img_files, size=min(ran_select, len(img_files)), replace=False)

        for img in random_imgs:
            img_path = os.path.join(path_i, img)
            img_array = imread(img_path)

            # Convert grayscale or RGBA images to RGB
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            # Crop the image
            img_cropped = img_array[18:(224-62), 50:(224-50)]

            # Flatten the image (no resizing)
            img_flattened = img_cropped.flatten()

            # Append the processed image data
            flat_data_arr.append(img_flattened)
            target_arr.append(i)

        print(f"Loaded category: {category} successfully")

    return np.array(flat_data_arr), np.array(target_arr)

# Load and preprocess the data
images, labels = load_data(path, CATEGORIES)

# Step 1: Fit the StandardScaler on the training data
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)  # Standardize the image data (mean = 0, std = 1)

# Step 2: Save the fitted scaler to 'scaler.joblib'
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")

# Step 3: Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images_scaled, labels, test_size=0.3, random_state=42
)
print(f"Training samples: {train_images.shape}")
print(f"Testing samples: {test_images.shape}")

# Step 4: Build and train the MLP model
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # Two hidden layers with 512 and 256 neurons
    activation='relu',              # Activation function
    solver='adam',                  # Optimizer
    max_iter=200,                   # Maximum iterations
    random_state=42,                # Random seed
    early_stopping=True,            # Enable early stopping
    n_iter_no_change=10             # Stop training if no improvement in 10 iterations
)

# Train the model
mlp_model.fit(train_images, train_labels)

# Step 5: Evaluate the model
test_accuracy = mlp_model.score(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")

# Step 6: Save the trained model
joblib.dump(mlp_model, joblib_file)
print(f"Trained model saved to {joblib_file}")

# Timing the process
end_time = time.time()