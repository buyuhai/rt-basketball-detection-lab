"""Build task 03 statistics and frame comparisons from A/B/C evaluation records."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
SETTINGS = (
    {"id": "A", "imgsz": 640, "conf": 0.10, "file": "A-640-conf010.jsonl"},
    {"id": "B", "imgsz": 320, "conf": 0.10, "file": "B-320-conf010.jsonl"},
    {"id": "C", "imgsz": 320, "conf": 0.25, "file": "C-320-conf025.jsonl"},
)
FOCUS_FRAMES = (25, 275)
RERUN = (
    "cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab && "
    ".venv/bin/python submission/task03/make_report.py"
)


def load():
    result = {}
    for setting in SETTINGS:
        lines = (RAW / setting["file"]).read_text(encoding="utf-8").splitlines()
        result[setting["id"]] = {
            row["frame"]: row for row in map(json.loads, lines) if row
        }
    return result


def summarize(rows):
    counts = {
        "ball_matches": 0, "ball_total": 0,
        "edge_matches": 0, "edge_total": 0,
        "regular_matches": 0, "regular_total": 0,
        "empty_false_alarms": 0, "empty_total": 0,
    }
    for row in rows.values():
        group = row["group"]
        if group == "empty":
            counts["empty_total"] += 1
            counts["empty_false_alarms"] += bool(row["detections"])
            continue
        counts["ball_total"] += 1
        counts["ball_matches"] += bool(row["matched"])
        if group == "edge":
            counts["edge_total"] += 1
            counts["edge_matches"] += bool(row["matched"])
        else:
            counts["regular_total"] += 1
            counts["regular_matches"] += bool(row["matched"])
    return counts


def draw_panel(image, row, title):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    annotation = row["annotation_xyxy"]
    if annotation:
        x1, y1, x2, y2 = (round(value) for value in annotation)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(canvas, "GT", (max(0, x1 + 2), max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1, cv2.LINE_AA)
    for detection in row["detections"]:
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection['confidence']:.3f} IoU {row['best_iou']:.3f}"
        cv2.putText(canvas, label, (max(0, x1 + 2), min(height - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1, cv2.LINE_AA)
    if not row["detections"]:
        cv2.putText(canvas, "no detection", (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 255), 2, cv2.LINE_AA)
    header = np.full((34, width, 3), 32, dtype=np.uint8)
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, canvas])


def make_comparison(frame, rows):
    image = cv2.imread(str(ROOT / "dataset-curated" / "images" / "val" /
                            f"rt-{frame:03d}.jpg"))
    if image is None:
        raise FileNotFoundError(f"缺少原图 rt-{frame:03d}.jpg，先运行 lab.py prepare")
    panels = []
    for setting in SETTINGS:
        row = rows[setting["id"]][frame]
        title = f"{setting['id']} | {setting['imgsz']} | conf {setting['conf']:.2f}"
        panels.append(draw_panel(image, row, title))
    separator = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
    body = np.hstack([panels[0], separator, panels[1], separator, panels[2]])
    banner = np.full((30, body.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(banner, f"frame {frame} | blue = GT, green = detection",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    composite = np.vstack([banner, body])
    path = OUT / f"frame-{frame:03d}-comparison.png"
    if not cv2.imwrite(str(path), composite):
        raise OSError(f"无法写图: {path}")


def main():
    data = load()
    stats = []
    for setting in SETTINGS:
        counts = summarize(data[setting["id"]])
        stats.append({
            "id": setting["id"],
            "imgsz": setting["imgsz"],
            "conf": setting["conf"],
            **counts,
        })

    summary = {
        "weights": "yolo11n.pt",
        "split": "val",
        "match_iou": 0.50,
        "annotation_source": "annotations.csv",
        "settings": stats,
        "recommendation": (
            "Keep 640/0.10 if preserving edge balls is the priority. "
            "If 320 is required, use 0.10, not 0.25: 0.25 loses frame 275 "
            "without recovering edge balls or reducing empty-frame false alarms."
        ),
        "rerun": RERUN,
    }
    (OUT / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (OUT / "per-frame.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "imgsz", "conf", "frame", "time_s", "group",
                         "annotation_xyxy", "detection_count", "best_iou",
                         "matched", "detections"])
        for setting in SETTINGS:
            for frame in sorted(data[setting["id"]]):
                row = data[setting["id"]][frame]
                writer.writerow([
                    setting["id"], setting["imgsz"], setting["conf"], frame,
                    row["time_s"], row["group"], row["annotation_xyxy"],
                    len(row["detections"]), row["best_iou"], row["matched"],
                    json.dumps(row["detections"], ensure_ascii=False),
                ])

    for frame in FOCUS_FRAMES:
        make_comparison(frame, data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
