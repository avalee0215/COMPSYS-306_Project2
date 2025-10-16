import os

# Set your dataset folder here
DATASET_PATH = 'dataset'  # folder containing subfolders like STOP, SPEED, etc.
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

def rename_images_in_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)])
    
    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"img{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        # Avoid overwriting if name already exists
        if src != dst:
            os.rename(src, dst)
    print(f"✅ Renamed {len(files)} images in {os.path.basename(folder_path)}")

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset folder '{DATASET_PATH}' not found!")
        return

    class_folders = [os.path.join(DATASET_PATH, d) for d in os.listdir(DATASET_PATH)
                     if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for class_folder in class_folders:
        rename_images_in_folder(class_folder)

    print("\n🎉 Renaming complete for all classes!")

if __name__ == "__main__":
    main()
