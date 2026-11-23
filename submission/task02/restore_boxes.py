"""Restore task 02 boxes from the CVAT Ultralytics YOLO export."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = OUT / "cvat-ultralytics-yolo.zip"
FILENAMES = ("rt-125.jpg", "rt-150.jpg", "rt-200.jpg", "plos02-055.jpg")
RESTORE = ("rt-150.jpg", "plos02-055.jpg")
RERUN = f"cd {ROOT} && .venv/bin/python submission/task02/restore_boxes.py"


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_image(filename):
    matches = list((ROOT / "dataset-curated" / "images").glob(f"*/{filename}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"无法唯一定位原图 {filename}: {matches}")
    return matches[0]


def read_references():
    references = {}
    with (ROOT / "annotations.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["x1"]:
                stem = f"{row['source']}-{int(row['frame']):03d}"
                references[stem] = [int(row[key]) for key in ("x1", "y1", "x2", "y2")]
    return references


def iou(box_a, box_b):
    a, b, c, d = box_a
    e, f, g, h = box_b
    intersect = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    union = (c-a)*(d-b) + (g-e)*(h-f) - intersect
    return intersect / union if union > 0 else 0.0


def decode(cx, cy, width, height, image_width, image_height):
    x1 = (cx - width / 2) * image_width
    y1 = (cy - height / 2) * image_height
    x2 = (cx + width / 2) * image_width
    y2 = (cy + height / 2) * image_height
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def draw_boxes(image, boxes):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, round(min(width, height) / 200))
    font_scale = max(.5, min(width, height) / 1200)
    for box in boxes:
        x1, y1, x2, y2 = box["pixel_xyxy"]
        p1 = (max(0, round(x1)), max(0, round(y1)))
        p2 = (min(width - 1, round(x2)), min(height - 1, round(y2)))
        cv2.rectangle(canvas, p1, p2, (0, 0, 0), thickness + 2)
        cv2.rectangle(canvas, p1, p2, (0, 255, 0), thickness)
        label = f"{box['class_id']} {box['class_name']}"
        origin = (max(0, round(x1)), max(12, round(y1) - 8))
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)
    return canvas


def main():
    references = read_references()
    expected_stems = [Path(name).stem for name in FILENAMES]
    records = []
    restored = []

    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        images_in_archive = any(
            name == filename or name.endswith("/" + filename)
            for filename in FILENAMES for name in names
        )
        data_yaml = yaml.safe_load(archive.read("data.yaml"))
        class_names = {int(key): str(value)
                       for key, value in data_yaml["names"].items()}
        if class_names != {0: "basketball"}:
            raise ValueError(f"类别映射不是 0: basketball: {class_names}")
        listed = [Path(line.strip()).stem for line in
                  archive.read("train.txt").decode("utf-8").splitlines() if line.strip()]
        if sorted(listed) != sorted(expected_stems):
            raise ValueError(f"train.txt 图片清单不符: {listed}")
        missing = [stem for stem in listed
                   if f"labels/train/{stem}.txt" not in names]
        if sorted(missing) != ["rt-200"]:
            raise ValueError(f"缺失标签的文件不是仅 rt-200: {missing}")

        for filename in FILENAMES:
            stem = Path(filename).stem
            path = source_image(filename)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            label_name = f"labels/train/{stem}.txt"
            label_present = label_name in names
            boxes = []
            if label_present:
                lines = [line for line in
                         archive.read(label_name).decode("utf-8").splitlines()
                         if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"标签行格式错误 {label_name}: {line}")
                    class_id = int(parts[0])
                    if class_id not in class_names:
                        raise ValueError(f"未知类别编号 {class_id}: {label_name}")
                    cx, cy, box_width, box_height = map(float, parts[1:])
                    boxes.append({
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "yolo_cxcywh": [round(cx, 8), round(cy, 8),
                                        round(box_width, 8), round(box_height, 8)],
                        "pixel_xyxy": decode(cx, cy, box_width, box_height, width, height),
                    })
            record = {
                "filename": filename,
                "source": relative(path),
                "width": width,
                "height": height,
                "has_label": bool(boxes),
                "label_state": "present" if boxes else ("empty" if label_present else "missing"),
                "label_file": label_name if label_present else None,
                "boxes": boxes,
            }
            if stem in references:
                reference = references[stem]
                record["reference_xyxy"] = reference
                record["reference_iou"] = round(
                    iou(reference, boxes[0]["pixel_xyxy"]) if boxes else 0.0, 4)
            records.append(record)

    for filename in RESTORE:
        record = next(item for item in records if item["filename"] == filename)
        path = ROOT / record["source"]
        image = cv2.imread(str(path))
        picture = OUT / f"{Path(filename).stem}-restored.png"
        if not cv2.imwrite(str(picture), draw_boxes(image, record["boxes"])):
            raise OSError(f"无法写图: {picture}")
        record["restored_image"] = relative(picture)
        restored.append({"image": record["filename"], "restored_image": relative(picture)})

    inventory = {
        "task": "task02-cvat-yolo-annotation-check",
        "archive": {
            "path": relative(ARCHIVE),
            "sha256": digest(ARCHIVE),
            "format": "CVAT Ultralytics YOLO Detection",
            "images_in_archive": images_in_archive,
        },
        "data_yaml": data_yaml,
        "class_map": {"0": "basketball"},
        "images": records,
        "restored": restored,
        "rerun": RERUN,
    }
    (OUT / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "class_map": inventory["class_map"],
        "images": [{key: item[key] for key in
                    ("filename", "width", "height", "has_label", "label_state", "boxes")}
                   for item in records],
        "restored": restored,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
