# Artifacts

这个目录只放整理后的轻量模型、CSV/JSON/YAML 结果和说明文档，不包含 Rawalk 原始图片、原始 3D pose、work-zone 原始图片或训练过程输出。

## 发布目录

| 目录 | 内容 |
|---|---|
| `rawalk_v1/` | Rawalk/EgoHumans 旧实验模型，文件名已重新整理 |
| `workzone_v1/` | work-zone 新实验模型、结果、20 图 JSON pipeline 输出和 Qwen prompt |
| `data/` | Rawalk/EgoHumans 旧实验处理后的 CSV/JSON/YAML 结果 |

Rawalk/EgoHumans 旧实验模型已整理到：

```text
artifacts/rawalk_v1/models/
```

work-zone 新实验模型已整理到：

```text
artifacts/workzone_v1/models/
```

## data

| 文件 | 用途 |
|---|---|
| `data/rawalk_ego_depth_all_step10.csv` | Rawalk ego 3D keypoints + calibration 生成的距离 GT，过滤前 |
| `data/rawalk_ego_depth_all_step10_m20.train.csv` | 过滤后的 train GT，`0.2m <= distance_m <= 20m` |
| `data/rawalk_ego_depth_all_step10_m20.eval.csv` | 过滤后的 eval GT，两个方案共用 |
| `data/rawalk_ego_depth_all_step10_m20.split.json` | 固定 train/eval split 摘要 |
| `data/rawalk_ego_scheme2_train_preds.csv` | 方案二 train split 的检测 + MoGe 匹配结果，用于训练 calibrator |
| `data/rawalk_ego_scheme2_eval_preds.csv` | 方案二 eval split 的检测 + MoGe 匹配结果 |
| `data/rawalk_ego_scheme2_eval_summary.json` | 方案二 eval 指标 |
| `data/rawalk_ego_scheme2_calibrator.metrics.json` | calibrator 训练/选择指标 |
| `data/yolo_distance_head_all_step10_m20.metrics.json` | 方案一 distance head 训练指标 |
| `data/rawalk_yolo11s_960_e20.metrics.json` | fine-tuned YOLO 检测指标 |
| `data/rawalk_yolo11s_960_e20_args.yaml` | fine-tuned YOLO 训练参数 |

注意：CSV 里的 `image_path` 仍然指向本机 Rawalk 原始图片路径。如果别人只下载 GitHub 仓库，可以查看数据处理结果和指标，也可以用 `artifacts/rawalk_v1/models/` 或 `artifacts/workzone_v1/models/` 跑新图片推理；要重新跑 Rawalk 原图评估，需要按 README 放置 Rawalk 原始数据。
