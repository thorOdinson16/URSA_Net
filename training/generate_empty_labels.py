import os

# Update these paths
image_dirs = [
    "r20_dataset/images/train",
    "r20_dataset/images/val",
    "r22_dataset/images/train",
    "r22_dataset/images/val",
    "r22_dataset/images/test"
]

label_dirs = [
    "r20_dataset/labels/train",
    "r20_dataset/labels/val",
    "r22_dataset/labels/train",
    "r22_dataset/labels/val",
    "r22_dataset/labels/test"
]

for img_dir, lbl_dir in zip(image_dirs, label_dirs):

    os.makedirs(lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img in images:
        name = os.path.splitext(img)[0]
        label_path = os.path.join(lbl_dir, name + ".txt")

        if not os.path.exists(label_path):
            open(label_path, "w").close()

print("Empty label files created where needed.")