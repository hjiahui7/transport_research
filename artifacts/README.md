# Artifacts

这个目录只放可以随代码一起上传的处理后 CSV/JSON/YAML 结果，不包含 Rawalk 原始图片、原始 3D pose、PM-HMCW 原始数据或模型权重。

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

注意：CSV 里的 `image_path` 仍然指向本机 Rawalk 原始图片路径。如果别人只下载 GitHub 仓库，可以复查数据处理结果和指标；要重新跑原图推理或加载模型，仍然需要按 README 放置 Rawalk 原始数据，并单独准备模型权重。
