"""Task 01: environment check, first inference, flip comparison, and frame replay."""
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy
import onnx
import onnxruntime
import torch
import torchvision
import ultralytics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lab import DATA, WEIGHTS, iou, load_detector, predict_frame, read_annotations

IMGSZ = 640
FIRST_CONF = 0.25
COMPARE_CONF = 0.10
CLASS = "sports ball"
VIDEO = ROOT / "data-src" / "rt.mp4"
RERUN = f"cd {ROOT} && .venv/bin/python submission/task01/task01.py"


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(model, target, image, confidence):
    detections, elapsed = predict_frame(model, target, image, IMGSZ, confidence)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    for detection in detections:
        detection["bbox_xyxy"] = [round(value, 2) for value in detection["bbox_xyxy"]]
    return detections, elapsed


def top_center(detections):
    if not detections:
        return None
    x1, y1, x2, y2 = detections[0]["bbox_xyxy"]
    return [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]


def annotate(image, detections):
    canvas = image.copy()
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
        if index == 0:
            center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
            cv2.circle(canvas, center, 4, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (255, 255, 255), 1, cv2.LINE_AA)
            label = f"{detection['class']} {detection['confidence']:.3f}"
            cv2.putText(canvas, label, (round(x1), max(round(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def environment_record():
    packages = {}
    for name, imported in (
        ("torch", torch.__version__), ("torchvision", torchvision.__version__),
        ("ultralytics", ultralytics.__version__), ("opencv-python", cv2.__version__),
        ("numpy", numpy.__version__), ("onnx", onnx.__version__),
        ("onnxruntime", onnxruntime.__version__),
    ):
        packages[name] = {"installed": metadata.version(name), "imported": imported}
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "packages": packages,
        "yolo_cli": str(Path(sys.executable).with_name("yolo")),
        "rerun": RERUN,
    }


def first_inference(model, target):
    path = DATA / "images" / "val" / "rt-100.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"缺少图片，先运行 lab.py prepare: {path}")
    detections, elapsed = predict(model, target, image, FIRST_CONF)
    height, width = image.shape[:2]
    picture = OUT / "rt-100-prediction.jpg"
    if not cv2.imwrite(str(picture), annotate(image, detections), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"无法写图: {picture}")
    annotation = next(row["box"] for row in read_annotations()
                      if row["source"] == "rt" and row["frame"] == 100)
    best_iou = iou(annotation, detections[0]["bbox_xyxy"]) if detections else 0.0
    record = {
        "task": "task01-first-inference",
        "image": relative(path),
        "frame": 100,
        "width": width,
        "height": height,
        "weights": {"path": relative(WEIGHTS), "sha256": digest(WEIGHTS)},
        "device": "cpu",
        "params": {"imgsz": IMGSZ, "conf": FIRST_CONF, "classes": [CLASS]},
        "detections": detections,
        "top_center_xy": top_center(detections),
        "annotation_xyxy": annotation,
        "top_iou": round(best_iou, 4),
        "elapsed_ms": elapsed,
        "annotated_image": relative(picture),
        "rerun": RERUN,
    }
    (OUT / "rt-100-inference.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def flip_comparison(model, target):
    path = DATA / "images" / "val" / "rt-225.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"缺少图片，先运行 lab.py prepare: {path}")
    flipped = cv2.flip(image, 1)
    flip_path = OUT / "rt-225-flip.png"
    if not cv2.imwrite(str(flip_path), flipped):
        raise OSError(f"无法写图: {flip_path}")
    flipped = cv2.imread(str(flip_path))
    if flipped is None:
        raise OSError(f"无法读图: {flip_path}")
    original_detections, original_ms = predict(model, target, image, COMPARE_CONF)
    flipped_detections, flipped_ms = predict(model, target, flipped, COMPARE_CONF)
    original_center = top_center(original_detections)
    flipped_center = top_center(flipped_detections)
    width = image.shape[1]
    mirror = None
    if original_center and flipped_center:
        expected = round(width - original_center[0], 2)
        mirror = {
            "expected_flipped_x": expected,
            "actual_flipped_x": flipped_center[0],
            "delta_x": round(flipped_center[0] - expected, 2),
        }
    confidence_comparison = {
        "original": original_detections[0]["confidence"] if original_detections else None,
        "flipped": flipped_detections[0]["confidence"] if flipped_detections else None,
        "equal": None,
        "difference": None,
    }
    if None not in (confidence_comparison["original"], confidence_comparison["flipped"]):
        confidence_comparison["equal"] = (
            confidence_comparison["original"] == confidence_comparison["flipped"]
        )
        confidence_comparison["difference"] = round(
            confidence_comparison["flipped"] - confidence_comparison["original"], 6
        )
    for name, canvas, detections in (
        ("rt-225-points.jpg", image, original_detections),
        ("rt-225-flip-points.png", flipped, flipped_detections),
    ):
        if not cv2.imwrite(str(OUT / name), annotate(canvas, detections)):
            raise OSError(f"无法写图: {OUT / name}")
    record = {
        "task": "task01-flip-comparison",
        "image": relative(path),
        "flipped_image": relative(flip_path),
        "flip_method": "cv2.flip(image, 1)",
        "device": "cpu",
        "params": {"imgsz": IMGSZ, "conf": COMPARE_CONF, "classes": [CLASS]},
        "original": {
            "width": width, "height": image.shape[0],
            "detections": original_detections,
            "top_center_xy": original_center,
            "elapsed_ms": original_ms,
            "annotated_image": relative(OUT / "rt-225-points.jpg"),
        },
        "flipped": {
            "width": flipped.shape[1], "height": flipped.shape[0],
            "detections": flipped_detections,
            "top_center_xy": flipped_center,
            "elapsed_ms": flipped_ms,
            "annotated_image": relative(OUT / "rt-225-flip-points.png"),
        },
        "mirror_check": mirror,
        "confidence_comparison": confidence_comparison,
        "local_cue_note": (
            "同一个卷积核在特征图上滑动，每个位置只对感受野内的局部像素做同一组"
            "加权求和；球翻到另一侧后，核在新位置提取镜像后的局部边缘和纹理线索。"
            "但卷积权重并非左右对称，翻转还会改变填充与网格对齐，所以两张有球图的"
            "最高分置信度不完全相同。"
        ),
        "rerun": RERUN,
    }
    (OUT / "rt-225-comparison.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def replay(model, target):
    reader = cv2.VideoCapture(str(VIDEO))
    if not reader.isOpened():
        raise FileNotFoundError(f"缺少视频，先运行 lab.py prepare: {VIDEO}")
    fps = reader.get(cv2.CAP_PROP_FPS)
    video_hash = digest(VIDEO)
    weights_hash = digest(WEIGHTS)
    wanted = (200, 225)
    records = []
    try:
        for frame in range(max(wanted) + 1):
            ok, image = reader.read()
            if not ok:
                raise RuntimeError(f"视频不足 {max(wanted) + 1} 帧，读取到第 {frame} 帧结束")
            if frame not in wanted:
                continue
            detections, elapsed = predict(model, target, image, COMPARE_CONF)
            height, width = image.shape[:2]
            records.append({
                "video": relative(VIDEO), "video_sha256": video_hash, "frame": frame,
                "time_s": round(frame / fps, 4) if fps > 0 else None,
                "fps": round(fps, 4), "width": width, "height": height,
                "weights": relative(WEIGHTS), "weights_sha256": weights_hash,
                "device": "cpu",
                "params": {"imgsz": IMGSZ, "conf": COMPARE_CONF, "classes": [CLASS]},
                "detections": detections, "top_center_xy": top_center(detections),
                "elapsed_ms": elapsed,
            })
    finally:
        reader.release()
    with (OUT / "replay-200-225.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main():
    torch.set_num_threads(3)
    environment = environment_record()
    (OUT / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model, target = load_detector(WEIGHTS)
    if model.names[target].lower() != CLASS:
        raise ValueError(f"权重中的目标类别不是 {CLASS}: {model.names[target]}")
    first = first_inference(model, target)
    comparison = flip_comparison(model, target)
    frames = replay(model, target)
    print(json.dumps({
        "environment": environment["packages"],
        "first_inference": first,
        "flip_comparison": {
            "original": comparison["original"]["top_center_xy"],
            "flipped": comparison["flipped"]["top_center_xy"],
            "mirror_check": comparison["mirror_check"],
            "confidence_comparison": comparison["confidence_comparison"],
        },
        "replay": [{
            "frame": record["frame"], "detections": record["detections"],
            "top_center_xy": record["top_center_xy"],
        } for record in frames],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
