import os
import shutil
import random

# PATHS
IMAGE_SRC = "DFUC2022_train_release/DFUC2022_train_images"
LABEL_SRC = "yolo_labels"

BASE_DIR = "yolo_dataset"

# OUTPUT FOLDERS
train_img_dir = os.path.join(BASE_DIR, "images/train")
val_img_dir = os.path.join(BASE_DIR, "images/val")
train_lbl_dir = os.path.join(BASE_DIR, "labels/train")
val_lbl_dir = os.path.join(BASE_DIR, "labels/val")

# CREATE FOLDERS
for path in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(path, exist_ok=True)

# GET FILES
images = os.listdir(IMAGE_SRC)

# SHUFFLE
random.shuffle(images)

# SPLIT (80% train, 20% val)
split_index = int(0.8 * len(images))

train_files = images[:split_index]
val_files = images[split_index:]

print(f"Total images: {len(images)}")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")

# COPY FUNCTION
def copy_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        img_src_path = os.path.join(IMAGE_SRC, file)
        lbl_file = file.replace(".jpg", ".txt")
        lbl_src_path = os.path.join(LABEL_SRC, lbl_file)

        # copy image
        shutil.copy(img_src_path, os.path.join(img_dest, file))

        # copy label (only if exists)
        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, os.path.join(lbl_dest, lbl_file))
        else:
            print(f"⚠️ Missing label for: {file}")

# COPY DATA
copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print("✅ YOLO dataset prepared successfully!")