# 任务 5：复核数据划分并做 Jackson 独立测试

## 两处来源交叉与修正

队友的划分有两处同源泄漏：

1. `rt-125` 被放进 train，其余 13 张 RT 帧在 val；RT 录像同时横跨 train/val。修正：`rt-125` 回到 val，RT 的 14 帧全部在 val。
2. `jackson-011` 被放进 val，其余 7 张 Jackson 帧在 test；Jackson 录像同时横跨 val/test。修正：`jackson-011` 回到 test，Jackson 的 8 帧全部在 test。

PLOS 六个研究录像全部在 train，没有交叉。修正前划分为 train 34 / val 14 / test 7；修正后为 **train 33 / val 14 / test 8**。`split-audit.json` 和 `split-by-source.csv` 记录了逐来源检查，修正后每个来源只出现在一个 split。

## 按视频来源整理的分组

帧号均为 0 基。

| 来源 | split | 帧数 | 帧号 | 有球/无球 |
|---|---|---:|---|---:|
| jackson | test | 8 | 003 011 019 027 035 043 051 059 | 7 / 1 |
| plos01 | train | 3 | 029 042 068 | 3 / 0 |
| plos02 | train | 6 | 003 016 055 068 081 094 | 4 / 2 |
| plos03 | train | 7 | 003 016 029 055 068 081 094 | 7 / 0 |
| plos06 | train | 5 | 003 029 042 055 094 | 4 / 1 |
| plos07 | train | 5 | 042 055 068 081 094 | 5 / 0 |
| plos09 | train | 7 | 003 016 029 042 055 068 081 | 7 / 0 |
| rt | val | 14 | 000 025 050 075 100 125 145 150 180 200 225 250 275 300 | 12 / 2 |

合计：train 33、val 14、test 8。Jackson 的 8 张图只用于测试，不参与任何训练或验证。

## 固定测试配置

配置在查看结果前写入 `test-config.json`：

- 权重：`yolo11n.pt`，SHA-256 `0ebbc80d…`
- 标注：`annotations.csv`，SHA-256 `336bcec3…`
- split：`test`，来源 `jackson`，8 帧
- `imgsz=416`、`conf=0.10`、`match_iou=0.50`
- 类别：`sports ball`；设备：CPU

复跑命令：

```bash
source /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab/.venv/bin/activate
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
python lab.py eval --split test --weights yolo11n.pt --imgsz 416 --conf 0.10 \
  --output runs/task05/jackson-416-conf010.jsonl
cp runs/task05/jackson-416-conf010.jsonl runs/task05/jackson-416-conf010.summary.json submission/task05/raw/
python submission/task05/make_report.py
```

## 8 张 Jackson 图的逐帧结果

| 帧 | 时间 s | 类型 | 人工框 | 检测 | 最高 IoU | 状态 | 多余框 | 需复核 |
|---:|---:|---|---|---|---:|---|---:|---|
| 003 | 0.1042 | small | `[584, 785, 694, 887]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 011 | 0.3819 | small | `[626, 1139, 720, 1233]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 019 | 0.6596 | small | `[605, 875, 707, 969]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 027 | 0.9374 | small | `[631, 879, 733, 981]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 035 | 1.2151 | small | `[694, 811, 805, 896]` | `[35.92, 1321.71, 97.30, 1394.76]`，`0.277` | 0.000 | 有球漏检 + 多余框 | 1 | 是 |
| 043 | 1.4929 | small | `[648, 495, 767, 597]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 051 | 1.7706 | edge | `[767, 0, 852, 73]` | 无 | 0.000 | 有球漏检 | 0 | 是 |
| 059 | 2.0484 | empty | 无 | 无 | 0.000 | 空帧正确 | 0 | 否 |

统计：

| 项目 | 计数 |
|---|---:|
| 正确匹配 | 0 / 7 张有球帧 |
| 有球漏检 | 7 / 7 张有球帧 |
| 无球误报 | 0 / 1 张无球帧 |
| 多余框 | 1 个（第 035 帧） |
| 需人工复核 | 7 / 8 帧（003、011、019、027、035、043、051） |

完整机器可读记录见 `per-frame.csv` 和 `summary.json`。

## 第 035 帧：有输出但不是有效匹配

第 035 帧是唯一有模型输出的图：

- 人工框：`[694, 811, 805, 896]`，框住球员手中的篮球。
- 模型输出：`[35.92, 1321.71, 97.30, 1394.76]`，置信度 `0.277`。
- 两者 IoU：`0.000`，低于 `0.50`，**不算有效匹配**。

输出落在画面左下角一名球员的绿色鞋上，而不是篮球上；该帧的真实篮球被漏检，同时多出一个鞋面误框。对照图：`jackson-035-check.png`（蓝框为人工篮球，绿框为模型输出）。

## 对这段录像的使用判断

在题面固定的 `yolo11n.pt + 416 + 0.10` 下，这段 Jackson 罚球录像的 7 张有球帧全部漏检，只有 1 个落在鞋上的多余框，无球帧没有误报。它不适合作为该配置下的自动回看/自动标注数据源；7 张有球帧都需要人工复核，第 059 帧可作为干净空帧。

8 帧样本很小，结论只针对这段录像和这个固定配置。如果之后要用于自动回看，需要提高输入尺寸、针对该机位微调，或改用更适合小目标的检测方案。

## 文件清单

- `split-by-source.csv`、`split-audit.json`：来源分组与两处交叉修正
- `test-config.json`：评测前固定的测试配置
- `per-frame.csv`：8 帧逐帧记录
- `summary.json`：匹配、漏检、误报、多余框和复核队列汇总
- `raw/jackson-416-conf010.jsonl`、`raw/jackson-416-conf010.summary.json`：原始评测记录
- `jackson-035-check.png`：唯一有输出帧的人工框/模型框对照
- `make_report.py`：逐帧表、汇总和对照图生成脚本
- `README.md`：本说明
