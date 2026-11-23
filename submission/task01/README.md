# 任务 1：环境核验、首次推理与翻转中心点对照

所有结果由 `task01.py` 生成；推理设备为 CPU，输入尺寸均为 `640`，类别过滤为 `sports ball`，中心点取最高分框的中心。

## 复跑入口

```bash
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
# 若 data-src/ 或 dataset-curated/ 不存在，先生成素材
.venv/bin/python lab.py prepare
# 生成环境记录、rt-100 结果、rt-225 翻转对照和回放记录
.venv/bin/python submission/task01/task01.py
```

ONNX 依赖安装命令：

```bash
.venv/bin/python -m pip install -c requirements.txt onnx==1.19.1 onnxruntime==1.32.0
```

## 环境核验

记录文件：`environment.json`。Python 路径与实测版本：

| 项目 | 值 |
|---|---|
| Python | `/home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab/.venv/bin/python`（3.12.3） |
| PyTorch | `2.6.0+cpu` |
| torchvision | `0.21.0+cpu` |
| Ultralytics | `8.3.232` |
| OpenCV | `opencv-python 4.11.0.86`（`cv2.__version__` 为 `4.11.0`） |
| NumPy | `1.26.4` |
| ONNX | `1.19.1` |
| ONNX Runtime | `1.32.0` |
| 设备 | CPU，`torch.cuda.is_available() == false` |
| `yolo` 命令 | `.venv/bin/yolo` |

`environment.json` 中每个包同时记录 `installed`（安装版本）和 `imported`（实际导入版本）；ONNX、ONNX Runtime 均已在 `.venv` 中正常导入，`pip check` 无冲突。

## rt-100 首次推理

- 输入：`dataset-curated/images/val/rt-100.jpg`（360×640）
- 权重：`yolo11n.pt`，SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- 设备：`cpu`
- 参数：`imgsz=640`、`conf=0.25`、`classes=["sports ball"]`
- 检测：`[139.41, 390.48, 260.47, 516.74]`，类别 `sports ball`，置信度 `0.5129`
- 最高分框中心：`(199.94, 453.61)`
- 与人工标注 `[136, 395, 255, 515]` 的 IoU：`0.8838`，框落在篮球上

记录：`rt-100-inference.json`；带框图：`rt-100-prediction.jpg`（绿框 + 洋红中心点）。

## rt-225 原图与左右翻转对照

原图为仓库生成的 `dataset-curated/images/val/rt-225.jpg`；`rt-225.jpg` 是它的逐字节副本，SHA-256 均为 `89395b3fededa39c042238b8d38c3087e77adc3e52177dfcedfa0be3baccad86`。翻转使用 `cv2.flip(image, 1)`，保存为 `rt-225-flip.png`，推理时重新读取该 PNG。

参数：`imgsz=640`、`conf=0.10`、`classes=["sports ball"]`，设备 CPU。

| 图像 | 最高分框 | 置信度 | 中心点（该图像像素坐标） |
|---|---|---|---|
| 原图 360×640 | `[24.39, 156.21, 133.00, 265.45]` | `0.2513` | `(78.69, 210.83)` |
| 翻转图 360×640 | `[227.55, 156.48, 336.28, 266.08]` | `0.2809` | `(281.91, 211.28)` |

镜像核对：`360 - 78.69 = 281.31`，翻转图实际中心 `281.91`，偏差 `+0.60 px`。球从画面左侧换到右侧。

记录：`rt-225-comparison.json`；带点原图：`rt-225-points.jpg`；带点翻转图：`rt-225-flip-points.png`。

同一个卷积核在特征图上滑动，每个位置只对其感受野内的局部像素做加权求和；图像翻转后，球移到新位置，核在新位置提取镜像后的局部边缘/纹理，因此仍能形成球特征。但卷积权重本身不是左右对称的，翻转还改变网格与填充的对齐，所以两张有球图的最高置信度不一样：`0.2513` 与 `0.2809`。

`rt-225-comparison.json` 的 `confidence_comparison` 记录两次最高分置信度、`equal: false` 和差值 `0.029589`；`local_cue_note` 保存上述卷积核解释。

## 第 200、225 帧回放

- 视频：`data-src/rt.mp4`，SHA-256 `f244ae13c9472ebb047d0c09caaaf41b947cad22decb66333c97fb82aff171a2`
- 参数：`imgsz=640`、`conf=0.10`、`classes=["sports ball"]`，设备 CPU
- 帧号从 0 开始，帧率 `60.0144`

记录：`replay-200-225.jsonl`，每行一个帧记录：

| 帧号 | 时间 | 检测结果 | 中心点 |
|---|---|---|---|
| 200 | 3.3325 s | `[]`（保留该帧记录） | `null` |
| 225 | 3.7491 s | `[24.64, 156.35, 132.91, 265.46]`，`sports ball`，`0.1937` | `(78.78, 210.90)` |

## 验收对照

- 使用项目 `.venv/bin/python`，Torch/torchvision/Ultralytics/OpenCV/NumPy/ONNX/ONNX Runtime 均可导入，版本见 `environment.json`。
- `rt-100` 的预测框与人工标注 IoU 为 `0.8838`，带框图中框落在篮球上。
- 回放记录按 `frame` 字段可复核；第 200 帧有记录且 `detections` 为空列表。
- 第 225 帧原图中心在左侧 `x=78.69`，翻转图中心在右侧 `x=281.91`；两者都可由记录中的检测框中心核对。

## 文件清单

- `task01.py`：复跑脚本
- `environment.json`：环境核验记录
- `rt-100-inference.json` / `rt-100-prediction.jpg`：首次推理记录与带框图
- `rt-225.jpg` / `rt-225-flip.png`：原图副本与左右翻转图（PNG）
- `rt-225-comparison.json`：中心点、检测框与镜像偏差对照
- `rt-225-points.jpg` / `rt-225-flip-points.png`：带中心点的原图与翻转图
- `replay-200-225.jsonl`：第 200、225 帧回放记录
