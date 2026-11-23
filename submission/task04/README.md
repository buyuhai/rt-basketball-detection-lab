# 任务 4：全量微调 vs 冻结前 10 层

## 结论先行

在固定的 640 输入、`conf=0.25`、IoU ≥ 0.50 口径下，两组结果完全相同：

| 组 | 有球匹配 | 完整球匹配 | 边缘球匹配 | 无球误报 |
|---|---:|---:|---:|---:|
| full | 0/12 | 0/10 | 0/2 | 0/2 |
| freeze10 | 0/12 | 0/10 | 0/2 | 0/2 |

两组都没有任何检测，训练期 val 的 P/R/mAP50/mAP50-95 也全为 0。因此现有结果**不支持判断哪一组更好**，结论是「暂时无法判断」。如果必须选一个继续尝试的方向，freeze10 的可训练参数更少（43.08% 参数被冻结），第 5 轮 val box/dfl loss 也略低；但这只是成本和损失信号，不是检测性能证据。

## 训练配置

两组从同一个 `yolo11n.pt` 出发，使用仓库 33 张训练图和 14 张验证图。训练参数沿用第三章：`imgsz=416`、5 轮、`batch=8`、AdamW、`lr0=0.001`、`mosaic=0`、`mixup=0`、`scale=0.20`、`seed=42`、`deterministic=True`、CPU、`workers=0`。

| 项目 | full | freeze10 |
|---|---|---|
| 权重 | `yolo11n.pt` | `yolo11n.pt` |
| freeze | 0 | 10 |
| 可训练参数 | 2,590,019 / 2,590,035 | 1,474,275 / 2,590,035 |
| 冻结参数 | 16（仅 DFL） | 1,115,760（前 10 层 + DFL） |
| 冻结比例 | 0.0006% | 43.08% |

Ultralytics 在两组中都会冻结 `.dfl`，所以这里的「全量微调」指除 DFL 外的全部参数可训练；两组唯一的配置差异是 `freeze`（以及输出目录名）。完整逐字段配置见 `full/args.yaml`、`freeze10/args.yaml` 和 `config-diff.json`。

## 哪些参数会更新

- **full**：除 DFL 的 16 个参数外，backbone、neck、检测头的参数都接收损失反向传播的梯度并更新；BN 的 running mean/var 也更新。
- **freeze10**：`model.0.` 至 `model.9.`（卷积、C3k2、SPPF 等前 10 层）的约 111.6 万参数冻结，不接收梯度；这些层的 BN 统计量也不更新。`model.10` 之后的 C2PSA、neck 和 Detect 头（除 DFL）共 1,474,275 个参数更新。

冻结范围审计见 `full/freeze-audit.json`、`freeze10/freeze-audit.json`。

## 训练集与验证集的作用

- **训练集（33 张）**：前向计算损失、反向传播并更新可训练参数。
- **验证集（14 张）**：不参与梯度更新；训练期间用于每轮监控和 `best.pt` 选择，任务最后又用同一批图做固定口径评测。
- 因此这里的最终指标不是独立测试集结果。要判断泛化能力，还需要另用 `test` 划分；当前结论只针对这 14 张验证图。

## 每轮训练记录

训练期的验证发生在 `imgsz=416`，使用 Ultralytics 默认验证参数，只用于监控；它**不是**下面的固定口径最终评测。

| 组 | epoch | train box | train cls | train dfl | val box | val cls | val dfl |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 1 | 2.1509 | 5.0252 | 0.9444 | 0.6871 | 3.6865 | 0.9411 |
| full | 2 | 2.3397 | 3.0887 | 1.0945 | 0.9768 | 4.1565 | 1.2454 |
| full | 3 | 2.4792 | 2.2966 | 1.0532 | 1.0776 | 4.2884 | 1.2920 |
| full | 4 | 2.0819 | 2.0282 | 0.9504 | 1.5575 | 4.4798 | 1.3915 |
| full | 5 | 1.8613 | 1.6409 | 0.8148 | 1.8504 | 4.6180 | 1.5202 |
| freeze10 | 1 | 2.1576 | 5.0722 | 0.9726 | 0.6517 | 3.6568 | 0.9525 |
| freeze10 | 2 | 2.3556 | 3.0871 | 1.1037 | 1.3944 | 4.0223 | 1.2041 |
| freeze10 | 3 | 2.2756 | 2.5055 | 1.0328 | 1.5783 | 4.1850 | 1.3065 |
| freeze10 | 4 | 2.0106 | 1.9902 | 0.9202 | 1.6100 | 4.4004 | 1.3632 |
| freeze10 | 5 | 1.8717 | 1.5790 | 0.7997 | 1.3856 | 4.5256 | 1.3093 |

两组合格训练期的 `metrics/precision(B)`、`recall(B)`、`mAP50(B)`、`mAP50-95(B)` 全部为 `0`。`best.pt` 的 `train_metrics` 与第 5 轮记录一致（fitness 为 0）；学习率、时间和完整列见 `per-epoch.csv` 与两组的 `results.csv`。

## 固定口径最终评测

`lab.py eval`：`split=val`、`imgsz=640`、`conf=0.25`、`match_iou=0.50`，只保留 `basketball` 类。评测权重为两组 `runs/task04/*/weights/best.pt`。

| 组 | 有球匹配 | 完整球匹配 | 边缘球匹配 | 无球误报 |
|---|---:|---:|---:|---:|
| full | 0/12 | 0/10 | 0/2 | 0/2 |
| freeze10 | 0/12 | 0/10 | 0/2 | 0/2 |

补充诊断：把阈值降到 `0.001` 后，两组在 14 张验证图上仍然没有任何检测；因此 0/12 不是 0.25 阈值单独造成的。该诊断不属于固定口径比较，原始记录在 `eval/diagnostic/`。

## 判断

现有证据只能说明：在这 33 张训练图、5 轮、1 类新检测头的设置下，两组都没有形成可用的检测响应。full 与 freeze10 在固定口径下完全打平，训练期的检测指标（P/R/mAP）也都为零，没有可用于区分的信号，因此**不能据此选择哪一组**，应给出「暂时无法判断」。

若必须继续试：freeze10 更省训练成本，且第 5 轮 val box/dfl loss 略低，可作为下一轮优先实验对象；但这不构成检测性能优势。更合理的下一步是增加轮数/训练数据，或先让检测头收敛，再用独立 test 集比较。

## 复跑入口

```bash
cd /home/taozhengzhe/rt_vision/homework4/rt-basketball-detection-lab
# 若数据/权重不存在，先运行 .venv/bin/python lab.py prepare
.venv/bin/python submission/task04/train_group.py --freeze 0 --name full
.venv/bin/python submission/task04/train_group.py --freeze 10 --name freeze10
.venv/bin/python lab.py eval --split val --weights runs/task04/full/weights/best.pt \
  --imgsz 640 --conf 0.25 --output runs/task04/eval/full-640-conf025.jsonl
.venv/bin/python lab.py eval --split val --weights runs/task04/freeze10/weights/best.pt \
  --imgsz 640 --conf 0.25 --output runs/task04/eval/freeze10-640-conf025.jsonl
# 将 runs/ 中的配置/结果同步到提交目录后运行：
.venv/bin/python submission/task04/make_report.py
```

## 文件清单

- `full/`：全量组的 `args.yaml`、`requested-config.json`、`freeze-audit.json`、`results.csv`
- `freeze10/`：冻结组的同四项记录
- `per-epoch.csv`：两组 5 轮合并记录
- `eval/full-640-conf025.jsonl`、`eval/full-640-conf025.summary.json`
- `eval/freeze10-640-conf025.jsonl`、`eval/freeze10-640-conf025.summary.json`
- `eval/diagnostic/`：0.001 阈值诊断记录，不属于固定口径比较
- `eval-summary.json`、`final-eval.csv`：固定口径统计
- `config-diff.json`：两组配置差异（仅 `freeze` 和输出名）
- `train_group.py`、`make_report.py`：训练与汇总脚本
- `README.md`：本说明
