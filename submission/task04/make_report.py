"""Combine task 04 training curves and fixed-setting evaluation records."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
GROUPS = ("full", "freeze10")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def eval_counts(rows):
    counts = {
        "ball_matches": 0, "ball_total": 0,
        "edge_matches": 0, "edge_total": 0,
        "regular_matches": 0, "regular_total": 0,
        "empty_false_alarms": 0, "empty_total": 0,
        "extra_boxes": 0,
    }
    for row in rows:
        group = row["group"]
        extra = len(row["detections"]) - int(row["matched"])
        counts["extra_boxes"] += extra
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


def main():
    with (OUT / "per-epoch.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header_written = False
        for group in GROUPS:
            with (OUT / group / "results.csv").open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if not header_written:
                    writer.writerow(["group", *reader.fieldnames])
                    header_written = True
                for row in reader:
                    writer.writerow([group, *[row[key] for key in reader.fieldnames]])

    final = {}
    for group in GROUPS:
        rows = load_jsonl(OUT / "eval" / f"{group}-640-conf025.jsonl")
        final[group] = {
            "weights": f"runs/task04/{group}/weights/best.pt",
            **eval_counts(rows),
        }
    summary = {
        "weights": "yolo11n.pt",
        "split": "val",
        "imgsz": 640,
        "conf": 0.25,
        "match_iou": 0.50,
        "groups": final,
        "decision": (
            "Cannot judge from the current evidence: both groups have zero matched "
            "ball frames at the fixed setting. Freeze10 has lower epoch-5 validation "
            "box/dfl loss but no detection above conf 0.001, so that is not a usable "
            "detection advantage."
        ),
    }
    (OUT / "eval-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (OUT / "final-eval.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "ball_matches", "ball_total", "regular_matches",
                         "regular_total", "edge_matches", "edge_total",
                         "empty_false_alarms", "empty_total", "extra_boxes"])
        for group in GROUPS:
            item = final[group]
            writer.writerow([group, item["ball_matches"], item["ball_total"],
                             item["regular_matches"], item["regular_total"],
                             item["edge_matches"], item["edge_total"],
                             item["empty_false_alarms"], item["empty_total"],
                             item["extra_boxes"]])

    configs = {group: json.loads((OUT / group / "requested-config.json").read_text())
               for group in GROUPS}
    keys = sorted(set(configs["full"]) | set(configs["freeze10"]))
    same = {key: configs["full"][key] for key in keys
            if configs["full"].get(key) == configs["freeze10"].get(key)}
    different = {key: [configs["full"].get(key), configs["freeze10"].get(key)]
                 for key in keys
                 if configs["full"].get(key) != configs["freeze10"].get(key)}
    (OUT / "config-diff.json").write_text(
        json.dumps({"same": same, "different": different}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
