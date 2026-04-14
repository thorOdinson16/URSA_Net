import os
import random
import shutil

# Paths
images_train = "dataset/images/train"
labels_train = "dataset/labels/train"

images_val = "dataset/images/val"
labels_val = "dataset/labels/val"

# Create validation folders
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# Percentage for validation
VAL_SPLIT = 0.15

# List all training images
images = [f for f in os.listdir(images_train) if f.endswith((".jpg", ".png"))]

# Shuffle for randomness
random.shuffle(images)

# Compute split size
val_count = int(len(images) * VAL_SPLIT)

val_images = images[:val_count]

for img in val_images:

    # Move image
    src_img = os.path.join(images_train, img)
    dst_img = os.path.join(images_val, img)
    shutil.move(src_img, dst_img)

    # Move label if it exists
    label_name = os.path.splitext(img)[0] + ".txt"
    src_label = os.path.join(labels_train, label_name)

    if os.path.exists(src_label):
        dst_label = os.path.join(labels_val, label_name)
        shutil.move(src_label, dst_label)

print(f"Validation split complete.")
print(f"Moved {val_count} images to validation set.")