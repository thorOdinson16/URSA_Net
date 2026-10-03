import os
import xml.etree.ElementTree as ET

xml_dir = "dataset/xml_annotations"
label_dir = "dataset/labels/train"

os.makedirs(label_dir, exist_ok=True)

class_map = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}

for xml_file in os.listdir(xml_dir):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        print("Skipping corrupted XML:", xml_file)
        continue

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    label_lines = []

    for obj in root.findall("object"):

        name_tag = obj.find("name")
        if name_tag is None:
            continue

        cls = name_tag.text.strip()

        # SAFE lookup (no KeyError possible)
        class_id = class_map.get(cls)

        if class_id is None:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )

    if len(label_lines) == 0:
        continue

    txt_file = xml_file.replace(".xml", ".txt")

    with open(os.path.join(label_dir, txt_file), "w") as f:
        f.write("\n".join(label_lines))

print("Conversion complete.")