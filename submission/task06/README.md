# 任务 6：固定 416 ONNX 与 PyTorch 后端核对

## 导出配置

| 项目 | 值 |
|---|---|
| 源权重 | `yolo11n.pt`，SHA-256 `0ebbc80d…` |
| ONNX | `yolo11n.onnx`，10,604,539 bytes，SHA-256 `5771f9fe…` |
| 格式 | ONNX |
| 输入尺寸 | `416 × 416`，batch 1，固定尺寸 |
| `dynamic` | `False` |
| `simplify` | `False` |
| `opset` | `17` |
| `half` | `False` |
| 设备 | CPU |
| 导出后输入张量 | `[1, 3, 416, 416]` |
| 导出后输出张量 | `[1, 84, 3549]` |

导出命令：

```bash
source /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab/.venv/bin/activate
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=416, dynamic=False, simplify=False, opset=17, batch=1, device='cpu', half=False)"
```

ONNX 文件位于本地仓库根目录，仓库 `.gitignore` 忽略 `*.onnx`；`export-config.json` 记录完整版本和哈希，可复现。

## 输入尺寸与预处理核对

`compare_onnx.py` 在推理前挂 PyTorch hook，并读取 ONNX Runtime session 的固定输入：

| 后端 | 实际输入尺寸 |
|---|---|
| PyTorch | `[1, 3, 416, 416]`（rt-100、rt-150、rt-200 三张均为同一尺寸） |
| ONNX Runtime | `[1, 3, 416, 416]` |

- ONNX 输入名：`images`，类型：`tensor(float)`
- `imgsz=416`、`rect=False`、`conf=0.10`、类别 `sports ball`
- 两个后端走同一条 Ultralytics predictor 预处理路径（相同缩放、letterbox 补边），输出框再映射回原图坐标
- 尺寸核对通过，记录见 `input-shape-record.json`

## 三张图逐图对照

| 帧 | PyTorch 框 / 类别 / 置信度 | ONNX Runtime 框 / 类别 / 置信度 | 空结果 |
|---:|---|---|---|
| 100 | `[140.13, 391.16, 260.04, 516.37]` / `sports ball` / `0.638961` | `[140.13, 391.16, 260.04, 516.37]` / `sports ball` / `0.638960` | 否 |
| 150 | 空 | 空 | 两边一致为空 |
| 200 | 空 | 空 | 两边一致为空 |

补充人工框对照（不作为后端切换的判定标准）：

- rt-100 人工框 `[136, 395, 255, 515]`，两边输出 IoU 均为 `0.8888`，属于有效检测。
- rt-150 人工框 `[0, 247, 31, 344]`，是边缘球；两边都漏检。
- rt-200 人工框为空，两边都输出空，正确。

## 配对与切换条件

| 帧 | 同类配对 | 配对 IoU | 置信度差 | 空结果一致 | 通过 |
|---:|---:|---:|---:|---|---|
| 100 | 1 对 | `1.000000` | `0.000000954` | 是 | 是 |
| 150 | 0 对 | — | — | 是（两边空） | 是 |
| 200 | 0 对 | — | — | 是（两边空） | 是 |

判定标准逐项核对：

- 同类一一配对：rt-100 的 1 个 `sports ball` 框一一配对；rt-150/200 两边都为空。
- 配对框 IoU：`1.000000` ≥ `0.95`。
- 置信度差：`0.000000954` ≤ `0.01`。
- 空结果一致：三帧全部一致。
- 实际输入尺寸：PyTorch 与 ONNX 均为 `[1, 3, 416, 416]`。

**结论：满足题面规定的切换后端条件，可以将固定 416 部署接口切到 ONNX Runtime。** 三张图中没有新增、消失或偏移的框；rt-100 两边框坐标完全相同，仅置信度有约 `1e-6` 的浮点差异。

需要注意：这个结论只说明两个后端在当前三张图上等价，不代表模型召回足够。rt-150 的边缘球在两边都漏检，这是模型在该尺寸下的召回问题，切换后端不会修复；回看工具仍应对边缘球保留人工复核。正样本只有 rt-100 一帧，证据规模也有限。

## 复跑入口

```bash
source /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab/.venv/bin/activate
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
python compare_onnx.py --onnx ./yolo11n.onnx --output submission/task06/backend-comparison.json
python submission/task06/make_report.py
```

`make_report.py` 会重新生成输入尺寸记录、逐图表、配对表和切换判断。

## 文件清单

- `export-config.json`：导出参数、版本、ONNX 哈希
- `input-shape-record.json`：两边实际输入尺寸、ONNX 输入名/类型、预处理说明
- `backend-comparison.json`：仓库核对入口的原始逐图输出
- `per-image.csv`：三张图 × 两后端的框、类别、置信度和空结果
- `pairing.csv`：同类配对、IoU、置信度差
- `summary.json`：配对指标和切换条件判定
- `make_report.py`：配对表和判定生成脚本
- `README.md`：本说明

## 选做：板卡部署

未执行：当前 WSL 没有 `~/.ssh/config` 中的板卡 Host，无法确认可用目标板卡；本机 `known_hosts` 也没有提供可用的板卡别名。主任务的 ONNX 固定尺寸与双后端核对不受影响。
