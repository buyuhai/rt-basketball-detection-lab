# Homework 4 提交索引

相关命令默认从仓库根目录执行，并先激活项目环境：

```bash
source /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab/.venv/bin/activate
```

| 任务 | 目录 | 主要内容 | 复跑入口 |
|---|---|---|---|
| 1 | `task01/` | 环境核验、rt-100 首次推理、rt-225 翻转中心点、200/225 回放 | `.venv/bin/python submission/task01/task01.py` |
| 2 | `task02/` | CVAT 导出包、四图核对、rt-150/plos02-055 还原图 | `.venv/bin/python submission/task02/restore_boxes.py` |
| 3 | `task03/` | 640/320 与 0.10/0.25 三组统计、25/275 三联对照 | 见 `task03/README.md`；`.venv/bin/python submission/task03/make_report.py` |
| 4 | `task04/` | full vs freeze10 训练记录、固定口径评测 | `.venv/bin/python submission/task04/train_group.py --freeze 0 --name full` 等，见 `task04/README.md` |
| 5 | `task05/` | 来源交叉修正、Jackson 8 图独立测试、035 帧对照 | `.venv/bin/python submission/task05/make_report.py` |
| 6 | `task06/` | 固定 416 ONNX、双后端尺寸/输出/切换判断 | `.venv/bin/python compare_onnx.py ...` 与 `submission/task06/make_report.py` |

说明：

- `.venv/`、`data-src/`、`dataset-curated/`、`runs/`、`*.pt`、`*.onnx` 按仓库 `.gitignore` 不进入 Git；数据、权重和 ONNX 可用 `lab.py prepare` 或 `task06/export-config.json` 中的命令重建。
- 任务 6 的 `yolo11n.onnx` 保留在本地仓库根目录，未提交；`task06/export-config.json` 记录了完整导出配置和哈希。
- CVAT 管理员密码保存在仓库外的 `~/cvat-admin-notes.txt`，不进入提交。
- `rt-125_150_200_02-055test.zip` 是本地初版 CVAT 导出，未提交；`task02/cvat-ultralytics-yolo.zip` 是最终修正版。
