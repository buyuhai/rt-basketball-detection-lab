# 任务 3：320 输入与阈值取舍

## 配置

三组都使用预训练 `yolo11n.pt`、仓库 `annotations.csv` 的 14 张验证图，匹配规则固定为 **IoU ≥ 0.50**，且只保留 `sports ball`。

| 组 | 输入尺寸 | 置信度阈值 | 相对变化 |
|---|---:|---:|---|
| A | 640 | 0.10 | 基准 |
| B | 320 | 0.10 | 相对 A 只变尺寸 |
| C | 320 | 0.25 | 相对 B 只变阈值 |

## 统计表

有球帧共 `12` 张（`10` 张完整球 + `2` 张边缘球），无球帧共 `2` 张（第 180、200 帧）。

| 组 | 有球匹配 | 边缘球匹配 | 无球误报 |
|---|---:|---:|---:|
| A：640 / 0.10 | **4/12** | **1/2** | 0/2 |
| B：320 / 0.10 | **3/12** | 0/2 | 0/2 |
| C：320 / 0.25 | **2/12** | 0/2 | 0/2 |

补充完整球（regular）计数：A `3/10`，B `3/10`，C `2/10`。分母均为对应类别帧数。

完整逐帧记录见 `per-frame.csv`；原始三组 JSONL 在 `raw/`。

## 重点帧

| 帧 | 类型 | A 640/0.10 | B 320/0.10 | C 320/0.25 |
|---:|---|---|---|---|
| 25 | 完整球 | 漏检 | 命中 `0.316`，IoU `0.956` | 命中 `0.316`，IoU `0.956` |
| 275 | 完整球 | 命中 `0.270`，IoU `0.973` | 命中 `0.156`，IoU `0.959` | 漏检 |
| 145 | 边缘球 | 命中 `0.125`，IoU `0.770` | 漏检 | 漏检 |
| 150 | 边缘球 | 漏检 | 漏检 | 漏检 |
| 180 | 无球 | 空 | 空 | 空 |
| 200 | 无球 | 空 | 空 | 空 |

对照图：

- `frame-025-comparison.png`：A 无检测，B/C 检出同一完整球。
- `frame-275-comparison.png`：A/B 检出，C 因 `0.25` 阈值漏检。

图中蓝框为标注 GT，绿框为检测；绿色文字为置信度和该帧最高 IoU。

## 变化说明

**完整球。** A 与 B 的 regular 命中数同为 `3/10`，但不是同一批帧：320/0.10 救回了第 25 帧，却在第 225 帧漏检，净结果持平。C 把阈值抬到 `0.25` 后，第 275 帧的 `0.156` 被切掉，regular 从 `3/10` 降到 `2/10`；第 25 帧的 `0.316` 仍保留。因此对完整球，`320/0.10` 可以与 `640/0.10` 打平，但 `320/0.25` 会净损失一帧。

**边缘球。** A 在 640 下以 `0.125` 命中第 145 帧；B 在 320 下完全没有该检测，C 也没有，边缘球从 `1/2` 降到 `0/2`。第 150 帧被三组全部漏检。降低阈值到 320/0.10 没有找回边缘球，说明这里的主要损失来自尺寸缩小，而不是 0.10 的阈值。

**无球帧。** 第 180、200 帧三组都为空，误报均为 `0/2`。在这组小样本上，`0.25` 没有带来任何误报收益，只减少了召回。

## 判断

**如果目标是尽量保住边缘球，不建议把回看工具降到 320，继续使用 A：640 / 0.10。** 320/0.10 会直接把边缘球从 `1/2` 降到 `0/2`，而完整球只是持平；320/0.25 又额外丢掉第 275 帧。

**如果工具必须使用 320，选 0.10（B），不要选 0.25（C）。** B 与 C 的边缘球都是 `0/2`、无球误报都是 `0/2`，但 C 的更高阈值只让第 275 帧完整球消失，没有换来边缘球或误报上的改善。

以上结论只代表这 14 张教学验证帧；样本很小，不能外推为通用阈值。

## 复跑

```bash
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
# 若数据集/权重不存在，先运行 .venv/bin/python lab.py prepare
.venv/bin/python lab.py eval --split val --weights yolo11n.pt --imgsz 640 --conf 0.10 --output runs/task03/A-640-conf010.jsonl
.venv/bin/python lab.py eval --split val --weights yolo11n.pt --imgsz 320 --conf 0.10 --output runs/task03/B-320-conf010.jsonl
.venv/bin/python lab.py eval --split val --weights yolo11n.pt --imgsz 320 --conf 0.25 --output runs/task03/C-320-conf025.jsonl
cp runs/task03/A-640-conf010.jsonl runs/task03/B-320-conf010.jsonl runs/task03/C-320-conf025.jsonl submission/task03/raw/
.venv/bin/python submission/task03/make_report.py
```

`make_report.py` 会重新生成 `stats.json`、`per-frame.csv` 和两张对照图。

## 文件清单

- `stats.json`：三组计数、分母、匹配规则、建议
- `per-frame.csv`：三组 × 14 帧的逐帧结果
- `raw/`：三组原始评测 JSONL
- `frame-025-comparison.png`：第 25 帧 A/B/C 三联对照
- `frame-275-comparison.png`：第 275 帧 A/B/C 三联对照
- `make_report.py`：统计表、逐帧 CSV 和对照图生成脚本
- `README.md`：本说明
