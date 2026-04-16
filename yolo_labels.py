import os
import cv2

# PATHS (NO CHANGE)
MASK_DIR = "DFUC2022_train_release/DFUC2022_train_masks"
IMAGE_DIR = "DFUC2022_train_release/DFUC2022_train_images"
LABEL_DIR = "yolo_labels"

os.makedirs(LABEL_DIR, exist_ok=True)

mask_files = os.listdir(MASK_DIR)

print("Generating YOLO labels...")

for mask_file in mask_files:
    mask_path = os.path.join(MASK_DIR, mask_file)
    image_path = os.path.join(IMAGE_DIR, mask_file.replace(".png", ".jpg"))

    mask = cv2.imread(mask_path, 0)

    # Threshold
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    img = cv2.imread(image_path)
    h_img, w_img, _ = img.shape

    # Convert to YOLO format
    x_center = (x + w/2) / w_img
    y_center = (y + h/2) / h_img
    w_norm = w / w_img
    h_norm = h / h_img

    label_file = os.path.join(LABEL_DIR, mask_file.replace(".png", ".txt"))

    with open(label_file, "w") as f:
        f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}")

print("✅ YOLO label generation completed!")