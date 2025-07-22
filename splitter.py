import os
import random
import shutil
import sklearn
from sklearn.model_selection import train_test_split

# --- Configuration ---
# IMPORTANT: Update these paths to match your project structure
SOURCE_DATA_DIR = r"C:\Users\CalidarTeam\OneDrive - Calidar Medical\Intern\Owen\BlockData_Processed\XRD_Data\All"
OUTPUT_DATASET_DIR = r"C:\Users\CalidarTeam\OneDrive - Calidar Medical\Intern\Owen\BlockData_Processed\dataset"

# The ratio for the split (e.g., 0.2 means 20% for validation, 80% for training)
VALIDATION_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

def split_yolo_dataset():
    """
    Splits the annotated YOLO dataset into training and validation sets.
    """
    print("--- Starting Dataset Split ---")

    # --- Step 1: Find all image files ---
    try:
        all_files = os.listdir(SOURCE_DATA_DIR)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"FATAL: No image files found in '{SOURCE_DATA_DIR}'. Halting.")
            return
        print(f"Found {len(image_files)} total images.")
    except FileNotFoundError:
        print(f"FATAL: Source data directory not found at '{SOURCE_DATA_DIR}'. Halting.")
        return

    # --- Step 2: Create a list of full paths ---
    image_paths = [os.path.join(SOURCE_DATA_DIR, f) for f in image_files]
    random.seed(RANDOM_STATE)
    random.shuffle(image_paths) # Shuffle to ensure randomness

    # --- Step 3: Split the list of image paths ---
    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_STATE
    )
    print(f"Splitting into {len(train_paths)} training images and {len(val_paths)} validation images.")

    # --- Step 4: Create the output directories ---
    train_img_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "labels")
    val_img_dir = os.path.join(OUTPUT_DATASET_DIR, "val", "images")
    val_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "val", "labels")

    # Clear old directories if they exist and create new ones
    shutil.rmtree(OUTPUT_DATASET_DIR, ignore_errors=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    print("Created YOLO directory structure.")

    # --- Step 5: Copy files to the new directories ---
    def copy_files(file_paths, img_dest_dir, lbl_dest_dir):
        """Copies images and their corresponding label files."""
        for img_path in file_paths:
            # Construct the corresponding label path
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            label_filename = base_filename + ".txt"
            label_path = os.path.join(SOURCE_DATA_DIR, label_filename)

            # Copy the image file
            shutil.copy(img_path, img_dest_dir)

            # Check for and copy the label file
            if os.path.exists(label_path):
                shutil.copy(label_path, lbl_dest_dir)
            else:
                print(f"WARNING: Label file not found for image: {os.path.basename(img_path)}")

    print("\nCopying training files...")
    copy_files(train_paths, train_img_dir, train_lbl_dir)
    print("Copying validation files...")
    copy_files(val_paths, val_img_dir, val_lbl_dir)

    print("\n--- Dataset Split Complete! ---")
    print(f"Your YOLO-formatted dataset is now ready in: {OUTPUT_DATASET_DIR}")


if __name__ == '__main__':
    split_yolo_dataset()
