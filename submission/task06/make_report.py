"""Build task 06 backend-pairing table and switch decision."""
import csv
import json
from pathlib import Path

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMPARISON = json.loads((OUT / "backend-comparison.json").read_text(encoding="utf-8"))
EXPECTED_SHAPE = [1, 3, 416, 416]


def iou(box_a, box_b):
    a, b, c, d = box_a
    e, f, g, h = box_b
    intersect = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    union = (c - a) * (d - b) + (g - e) * (h - f) - intersect
    return intersect / union if union > 0 else 0.0


def load_annotations():
    result = {}
    with (ROOT / "annotations.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source"] == "rt" and int(row["frame"]) in (100, 150, 200):
                frame = int(row["frame"])
                result[frame] = [int(row[key]) for key in ("x1", "y1", "x2", "y2")] if row["x1"] else []
    return result


def make_pairs(pytorch, onnx):
    pairs = []
    pytorch_left = set(range(len(pytorch)))
    onnx_left = set(range(len(onnx)))
    while True:
        candidates = []
        for i in pytorch_left:
            for j in onnx_left:
                if pytorch[i]["class"] == onnx[j]["class"]:
                    candidates.append((iou(pytorch[i]["bbox_xyxy"], onnx[j]["bbox_xyxy"]), i, j))
        if not candidates:
            break
        score, i, j = max(candidates)
        p, o = pytorch[i], onnx[j]
        diff = abs(p["confidence"] - o["confidence"])
        pairs.append({
            "pytorch": p, "onnxruntime": o, "class": p["class"],
            "iou": round(score, 6), "confidence_diff": round(diff, 9),
            "paired": True,
            "pass": score >= 0.95 and diff <= 0.01,
        })
        pytorch_left.remove(i)
        onnx_left.remove(j)
    for i in sorted(pytorch_left):
        pairs.append({"pytorch": pytorch[i], "onnxruntime": None, "class": pytorch[i]["class"],
                      "iou": 0.0, "confidence_diff": None, "paired": False, "pass": False})
    for j in sorted(onnx_left):
        pairs.append({"pytorch": None, "onnxruntime": onnx[j], "class": onnx[j]["class"],
                      "iou": 0.0, "confidence_diff": None, "paired": False, "pass": False})
    return pairs


def max_iou(boxes, annotation):
    return round(max((iou(annotation, box["bbox_xyxy"]) for box in boxes), default=0.0), 4)


def main():
    annotations = load_annotations()
    session = ort.InferenceSession(str(ROOT / "yolo11n.onnx"),
                                   providers=["CPUExecutionProvider"])
    onnx_input = session.get_inputs()[0]
    expected_match = (
        COMPARISON["onnx_input_shape"] == EXPECTED_SHAPE and
        COMPARISON["pytorch_input_shapes"] == [EXPECTED_SHAPE] * len(COMPARISON["frames"])
    )
    input_record = {
        "imgsz": COMPARISON["imgsz"],
        "conf": COMPARISON["conf"],
        "rect": COMPARISON["rect"],
        "classes": ["sports ball"],
        "pytorch_input_shapes": COMPARISON["pytorch_input_shapes"],
        "onnx_input_shape": COMPARISON["onnx_input_shape"],
        "onnx_input_name": onnx_input.name,
        "onnx_input_type": onnx_input.type,
        "expected_shape": EXPECTED_SHAPE,
        "match": expected_match,
        "preprocessing": (
            "Both backends use the same Ultralytics predictor path with imgsz=416 "
            "and rect=False; images are letterboxed with the same scale and padding "
            "by compare_onnx.py, so boxes are mapped back to original image coordinates."
        ),
        "source": "submission/task06/backend-comparison.json",
    }
    (OUT / "input-shape-record.json").write_text(
        json.dumps(input_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    frames = []
    with (OUT / "per-image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "backend", "annotation_xyxy", "boxes",
                         "classes", "confidences", "empty"])
        for item in COMPARISON["frames"]:
            frame = item["frame"]
            annotation = annotations.get(frame, [])
            pairs = make_pairs(item["pytorch"], item["onnxruntime"])
            empty_consistent = (not item["pytorch"]) == (not item["onnxruntime"])
            if not item["pytorch"] and not item["onnxruntime"]:
                frame_pass = True
            elif not item["pytorch"] or not item["onnxruntime"]:
                frame_pass = False
            else:
                frame_pass = bool(pairs) and all(pair["pass"] for pair in pairs)
            record = {
                "frame": frame,
                "annotation_xyxy": annotation,
                "pytorch": item["pytorch"],
                "onnxruntime": item["onnxruntime"],
                "pairs": pairs,
                "empty_consistent": empty_consistent,
                "gt_iou_pytorch": max_iou(item["pytorch"], annotation) if annotation else None,
                "gt_iou_onnxruntime": max_iou(item["onnxruntime"], annotation) if annotation else None,
                "pass": frame_pass,
            }
            frames.append(record)
            for backend, boxes in (("pytorch", item["pytorch"]), ("onnxruntime", item["onnxruntime"])):
                writer.writerow([
                    frame, backend, annotation,
                    json.dumps([box["bbox_xyxy"] for box in boxes]),
                    json.dumps([box["class"] for box in boxes]),
                    json.dumps([box["confidence"] for box in boxes]),
                    not boxes,
                ])

    positive_pairs = [pair for frame in frames for pair in frame["pairs"] if pair["paired"]]
    summary = {
        "input_shape_check": input_record,
        "frames": frames,
        "switch_condition": {
            "paired_one_to_one_same_class": all(
                not frame["pytorch"] and not frame["onnxruntime"] or
                (frame["pairs"] and all(pair["paired"] for pair in frame["pairs"]))
                for frame in frames
            ),
            "min_paired_iou": min((pair["iou"] for pair in positive_pairs), default=None),
            "max_confidence_diff": max((pair["confidence_diff"] for pair in positive_pairs), default=None),
            "iou_at_least_0_95": all(pair["iou"] >= 0.95 for pair in positive_pairs),
            "confidence_diff_at_most_0_01": all(pair["confidence_diff"] <= 0.01 for pair in positive_pairs),
            "empty_results_consistent": all(frame["empty_consistent"] for frame in frames),
            "ready": all(frame["pass"] for frame in frames) and expected_match,
        },
        "rerun": "cd " + str(ROOT) + " && .venv/bin/python submission/task06/make_report.py",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "pairing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "pytorch_box", "onnx_box", "class", "iou",
                         "conf_pytorch", "conf_onnx", "conf_diff", "paired", "pass"])
        for frame in frames:
            for pair in frame["pairs"]:
                writer.writerow([
                    frame["frame"],
                    pair["pytorch"]["bbox_xyxy"] if pair["pytorch"] else None,
                    pair["onnxruntime"]["bbox_xyxy"] if pair["onnxruntime"] else None,
                    pair["class"], pair["iou"],
                    pair["pytorch"]["confidence"] if pair["pytorch"] else None,
                    pair["onnxruntime"]["confidence"] if pair["onnxruntime"] else None,
                    pair["confidence_diff"], pair["paired"], pair["pass"],
                ])
    print(json.dumps(summary["switch_condition"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
