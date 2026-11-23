"""Build task 05 Jackson test table, review summary and one annotated example."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw" / "jackson-416-conf010.jsonl"


def classify(row):
    annotation = row["annotation_xyxy"]
    detections = row["detections"]
    extra = len(detections) - int(row["matched"])
    if annotation and row["matched"]:
        status = "正确匹配"
        review = bool(extra)
    elif annotation:
        status = "有球漏检" + (" + 多余框" if detections else "")
        review = True
    elif detections:
        status = "无球误报"
        review = True
    else:
        status = "空帧正确"
        review = False
    return status, extra, review


def draw_example(row):
    image = cv2.imread(str(ROOT / "dataset-curated" / "images" / "test" /
                            f"jackson-{row['frame']:03d}.jpg"))
    if image is None:
        raise FileNotFoundError(f"缺少 jackson-{row['frame']:03d}.jpg")
    canvas = image.copy()
    annotation = row["annotation_xyxy"]
    if annotation:
        x1, y1, x2, y2 = annotation
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(canvas, "GT ball", (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    detection = row["detections"][0]
    x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(canvas, f"pred {detection['confidence']:.3f}",
                (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    banner = np.full((54, canvas.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(banner,
                f"frame {row['frame']:03d} | blue = GT ball | green = pred "
                f"{detection['confidence']:.3f} | IoU {row['best_iou']:.3f} (not a match)",
                (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                cv2.LINE_AA)
    picture = OUT / f"jackson-{row['frame']:03d}-check.png"
    if not cv2.imwrite(str(picture), np.vstack([banner, canvas])):
        raise OSError(f"无法写图: {picture}")
    return picture


def main():
    config = json.loads((OUT / "test-config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    records = []
    for row in rows:
        status, extra, review = classify(row)
        records.append({
            "frame": row["frame"],
            "time_s": row["time_s"],
            "group": row["group"],
            "annotation_xyxy": row["annotation_xyxy"],
            "detections": row["detections"],
            "detection_count": len(row["detections"]),
            "best_iou": row["best_iou"],
            "matched": row["matched"],
            "status": status,
            "extra_boxes": extra,
            "needs_review": review,
        })

    with (OUT / "per-frame.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "group", "annotation_xyxy",
                         "detection_count", "detections", "best_iou", "matched",
                         "status", "extra_boxes", "needs_review"])
        for record in records:
            writer.writerow([
                record["frame"], record["time_s"], record["group"],
                record["annotation_xyxy"], record["detection_count"],
                json.dumps(record["detections"], ensure_ascii=False),
                record["best_iou"], record["matched"], record["status"],
                record["extra_boxes"], record["needs_review"],
            ])

    ball_rows = [record for record in records if record["annotation_xyxy"]]
    empty_rows = [record for record in records if not record["annotation_xyxy"]]
    summary = {
        "config": config,
        "frames": len(records),
        "ball_frames": len(ball_rows),
        "empty_frames": len(empty_rows),
        "correct_matches": sum(record["matched"] for record in ball_rows),
        "ball_misses": sum(not record["matched"] for record in ball_rows),
        "empty_false_alarms": sum(bool(record["detections"]) for record in empty_rows),
        "empty_correct": sum(not record["detections"] for record in empty_rows),
        "extra_boxes": sum(record["extra_boxes"] for record in records),
        "frames_needing_review": sum(record["needs_review"] for record in records),
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "review_frames": [record["frame"] for record in records if record["needs_review"]],
        "detection_frames": [record["frame"] for record in records if record["detections"]],
    }
    with_detection = [record for record in records if record["detections"]]
    if with_detection:
        example = max(with_detection, key=lambda record: record["detections"][0]["confidence"])
        picture = draw_example(example)
        summary["example"] = {
            "frame": example["frame"],
            "annotation_xyxy": example["annotation_xyxy"],
            "detection": example["detections"][0],
            "best_iou": example["best_iou"],
            "matched": example["matched"],
            "image": str(picture.relative_to(ROOT)),
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
