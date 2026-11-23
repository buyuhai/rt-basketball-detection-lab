# 任务 2：CVAT YOLO 标注核对与坐标还原

## 当前状态

CVAT Community **v2.83.0** 已部署并验证。已在 CVAT 中创建任务 `task02-basketball`（task id 1），上传四张图，按可见球边缘标注，并导出 **Ultralytics YOLO Detection 1.0**。

导出包已替换为此前偏移版本的修正版：三张有球图的导出框与仓库参考框 IoU 均为 `1.0`，`rt-200` 保持无标签。初版导出仅保留在本地 `rt-125_150_200_02-055test.zip`，提交目录使用修正版。

## CVAT 环境

- URL：`http://localhost:8080`
- CVAT：`2.83.0`
- Docker Engine：`29.10.0`
- Docker Compose：`v5.8.0`
- 源码目录：`/home/taozhengzhe/cvat`（git tag `v2.83.0`）
- 管理员：`admin`；密码与部署记录在 `~/cvat-admin-notes.txt`（权限 600，不放入仓库）
- 下次启动命令：

```bash
cd /home/taozhengzhe/cvat && docker compose up -d
```

如果当前 shell 还没有 docker 组权限，先执行 `newgrp docker` 或重新登录一次；若 Docker 守护进程未启动，先执行 `sudo systemctl start docker`。

本机 shell 设置了 `HTTP_PROXY/HTTPS_PROXY=127.0.0.1:12450`，直接 `curl http://localhost:8080` 会返回代理的 502。本地检查请使用：

```bash
curl --noproxy '*' http://localhost:8080/api/server/about
```

## 复跑入口

```bash
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
# 若 dataset-curated/ 或 data-src/ 不存在，先生成素材
.venv/bin/python lab.py prepare
# 读取 submission/task02/cvat-ultralytics-yolo.zip，生成 inventory.json 和两张还原图
.venv/bin/python submission/task02/restore_boxes.py
```

如需重新导出，在 CVAT 任务 1 中调整后导出，覆盖 `submission/task02/cvat-ultralytics-yolo.zip`，再运行上面的还原命令。

## 导出包核对

- 提交包：`cvat-ultralytics-yolo.zip`
- SHA-256：`a1656280f89609abac5e617664d426d3aa66e32163b5374de1a437f9d13d38dc`
- 格式：CVAT **Ultralytics YOLO Detection 1.0**
- `data.yaml`：`names: {0: basketball}`，`train: train.txt`
- `train.txt` 列出四张图：`plos02-055.jpg`、`rt-125.jpg`、`rt-150.jpg`、`rt-200.jpg`
- 包内含 `images/train/` 四张原图和 `labels/train/` 三个标签文件
- `rt-200` 没有标签文件；`inventory.json` 记为 `has_label: false`、`label_state: "missing"`、`boxes: []`

类别与文件名对应正常：标签首列均为 `0`，对应 `basketball`；标签文件名与图片 stem 一致。

## 四张图核对记录

完整字段见 `inventory.json`。

| 文件名 | 宽×高 | 有无标签 | 导出框 `x1,y1,x2,y2` | 与参考框 IoU |
|---|---:|---|---:|---:|
| `rt-125.jpg` | 360×640 | 有 | `[48.00, 303.00, 167.00, 425.00]` | `1.0000` |
| `rt-150.jpg` | 360×640 | 有 | `[0.00, 247.00, 31.00, 344.00]` | `1.0000` |
| `rt-200.jpg` | 360×640 | 无（标签缺省） | `[]` | — |
| `plos02-055.jpg` | 1920×1080 | 有 | `[1072.00, 563.00, 1141.00, 632.00]` | `1.0000` |

`rt-150` 只框画面内可见部分，左边界为 `x=0`；`rt-200` 保留在图片清单和记录中，标签为空/缺省。

## 坐标还原

对每个标签行 `0 cx cy w h`，使用**该图自己的**宽高 `W×H`：

```text
x1 = (cx - w/2) * W
y1 = (cy - h/2) * H
x2 = (cx + w/2) * W
y2 = (cy + h/2) * H
```

- `rt-150` 使用 `360×640`；
- `plos02-055` 使用 `1920×1080`。

还原图：

- `rt-150-restored.png`：`360×640`
- `plos02-055-restored.png`：`1920×1080`

绿框上的文字 `0 basketball` 同时核对了类别编号与类别名。

## 文件清单

- `cvat-ultralytics-yolo.zip`：CVAT 修正版导出包（含图片与标签）
- `inventory.json`：四张图的文件名、宽高、标签状态、YOLO 坐标与像素框
- `rt-150-restored.png`：rt-150 原图尺寸下的框还原
- `plos02-055-restored.png`：plos02-055 原图尺寸下的框还原
- `restore_boxes.py`：解析导出包并生成还原图/核对记录
- `README.md`：本说明
