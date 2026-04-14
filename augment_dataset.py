import os
import cv2
import albumentations as A
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
INPUT_DATASETS = [
    ("r20_dataset/images/train", "r20_dataset/labels/train"),
    ("r22_dataset/images/train", "r22_dataset/labels/train"),
]

OUTPUT_IMAGES_DIR = "augmented_output/images"
OUTPUT_LABELS_DIR = "augmented_output/labels"

NUM_AUGMENTATIONS_PER_IMAGE = 1

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

IMAGE_EXT = {".jpg",".jpeg",".png",".bmp"}


# ─────────────────────────────
# AUGMENTATION PIPELINE
# (all transforms randomised)
# ─────────────────────────────
def build_pipeline():

    return A.Compose(

        [

            A.OneOf([
                A.MotionBlur(blur_limit=(3,15)),
                A.GaussianBlur(blur_limit=(3,7))
            ], p=0.5),

            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),

            A.CLAHE(p=0.4),

            A.GaussNoise(p=0.5),

            A.Perspective(scale=(0.02,0.08), p=0.4),

            A.Rotate(limit=15,
                     border_mode=cv2.BORDER_REFLECT_101,
                     p=0.5),

            A.RandomRain(
                drop_length=20,
                drop_width=1,
                brightness_coefficient=0.8,
                rain_type="drizzle",
                p=0.3
            ),

            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.3
            ),

            A.OneOf([
                A.Sharpen(alpha=(0.1,0.4)),
                A.GaussianBlur(blur_limit=(3,7))
            ], p=0.3),

        ],

        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2
        )

    )


# ─────────────────────────────
# LABEL IO
# ─────────────────────────────
def read_labels(path):

    bboxes = []
    classes = []

    if not os.path.exists(path):
        return bboxes, classes

    with open(path) as f:

        for line in f:

            parts = line.split()

            if len(parts) != 5:
                continue

            classes.append(int(parts[0]))
            bboxes.append(list(map(float,parts[1:])))

    return bboxes, classes


def write_labels(path,bboxes,classes):

    with open(path,"w") as f:

        for c,b in zip(classes,bboxes):

            x,y,w,h = b

            x = max(0,min(1,x))
            y = max(0,min(1,y))
            w = max(0,min(1,w))
            h = max(0,min(1,h))

            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────
# PROCESS SINGLE IMAGE
# ─────────────────────────────
def process_image(args):

    img_path, lbl_dir = args

    pipeline = build_pipeline()

    stem = img_path.stem
    dataset_name = Path(img_path).parts[0]

    label_path = os.path.join(lbl_dir,stem+".txt")

    image = cv2.imread(str(img_path))

    if image is None:
        return

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    bboxes,classes = read_labels(label_path)

    try:

        result = pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=classes
        )

    except:
        return

    aug_img = result["image"]
    aug_boxes = result["bboxes"]
    aug_classes = result["class_labels"]

    new_name = f"{stem}_{dataset_name}_aug0"

    out_img = os.path.join(OUTPUT_IMAGES_DIR,new_name+".jpg")
    out_lbl = os.path.join(OUTPUT_LABELS_DIR,new_name+".txt")

    cv2.imwrite(out_img,cv2.cvtColor(aug_img,cv2.COLOR_RGB2BGR))

    write_labels(out_lbl,aug_boxes,aug_classes)


# ─────────────────────────────
# BUILD IMAGE LIST
# ─────────────────────────────
def collect_images():

    jobs = []

    for img_dir,lbl_dir in INPUT_DATASETS:

        for img_path in Path(img_dir).iterdir():

            if img_path.suffix.lower() in IMAGE_EXT:

                jobs.append((img_path,lbl_dir))

    return jobs


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():

    jobs = collect_images()

    print("Total images:",len(jobs))

    workers = multiprocessing.cpu_count()

    print("Using",workers,"CPU cores")

    with ProcessPoolExecutor(max_workers=workers) as exe:

        list(tqdm(exe.map(process_image, jobs), total=len(jobs)))

    print("Augmentation complete.")


if __name__ == "__main__":

    main()