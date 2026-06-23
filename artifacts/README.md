# Artifacts

这个目录只放处理后的 CSV/JSON/YAML 结果，不包含 Rawalk 原始图片、原始 3D pose 或训练过程输出。

小模型权重已经放在仓库根目录的 `models/`：

```text
models/yolo11n.pt
models/yolo_distance_head_all_step10_m20.pt
models/rawalk_yolo11s_960_e20_best.pt
models/rawalk_ego_scheme2_calibrator.joblib
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

注意：CSV 里的 `image_path` 仍然指向本机 Rawalk 原始图片路径。如果别人只下载 GitHub 仓库，可以查看数据处理结果和指标，也可以用 `models/` 跑新图片推理；要重新跑 Rawalk 原图评估，需要按 README 放置 Rawalk 原始数据。
