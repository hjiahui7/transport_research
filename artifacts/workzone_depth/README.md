# Work-Zone 测距

三条路线共用官方 YOLO11n 的 `person` 检测框。数据包含 1,685 张图和 3,278 个工人标注。固定切分使用 1,435 张训练图和 250 张验证图。

| 文件 | 用途 |
|---|---|
| [`models/yolo11n_distance_head.pt`](models/yolo11n_distance_head.pt) | 官方 YOLO11n 特征上的测距头 |
| [`models/moge_calibrator.joblib`](models/moge_calibrator.joblib) | MoGe 的 MLP 距离校准器 |
| [`comparison.csv`](comparison.csv) | 三条路线的验证指标 |
| [`zone_confusion_matrices.csv`](zone_confusion_matrices.csv) | 各路线的分区结果，单列漏检与缺失距离 |

距离误差统一用 336 个有数值真值且匹配成功的工人计算。分区准确率以全部 415 个工人为分母，包含人工分区标注，漏检计为未分对。

## 单图推理

在仓库根目录运行。测试环境为 Python 3.12、PyTorch 2.11 CUDA 12.8、Ultralytics 8.4.55 和 scikit-learn 1.8.0。另需 OpenCV、Pillow、NumPy、joblib。MoGe 路线还需安装 [MoGe](https://github.com/microsoft/MoGe) 并取得 `Ruicheng/moge-2-vits-normal` 权重。

仓库已有的 `artifacts/rawalk_v1/models/rawalk_scheme1_yolo11n_base.pt` 与本次实验使用的官方 `yolo11n.pt` 字节一致。

**YOLO Distance Head**

```bash
python -m human_detect.infer_distance_head --image path/to/image.png --checkpoint artifacts/workzone_depth/models/yolo11n_distance_head.pt --imgsz 960 --device cuda:0 --no-half --out runs/workzone_depth/head.json
```

**MoGe + MLP 校准**

```bash
python -m human_detect.infer --image path/to/image.png --detector artifacts/rawalk_v1/models/rawalk_scheme1_yolo11n_base.pt --geometry-model Ruicheng/moge-2-vits-normal --imgsz 960 --geom-size 640 --num-tokens 1200 --device cuda:0 --no-detector-half --calibrator artifacts/workzone_depth/models/moge_calibrator.joblib --out runs/workzone_depth/moge.json
```

MoGe 的 JSON 同时包含原始 `distance_m` 和校准后的 `distance_calibrated_m`。本机已有 MoGe 权重时，可将 `--geometry-model` 设为 `model.pt` 的路径。

## 复现训练与评估

准备同版本的 Work-Zone RGB 图像与标注目录。Wave 1 使用 `annotations/worker_gt_merged.csv` 和 `images/`，Wave 2 使用 `workzone_rgbd_dataset_wave2/annotations.csv` 和对应图像。训练需要 `lidar_distance_gt_m` 与 `representative_lidar_z_m`。`lidar_distance_gt_m` 是采样点到相机的三维直线距离中位数，缺失数值时保留人工分区。此目录提供模型和汇总指标，逐人标注和原图需单独取得。

```bash
python -m human_detect.prepare_workzone_v2 --dataset-root path/to/work-zone-safety-rgbd-dataset
python artifacts/workzone_depth/run_experiment.py --moge path/to/model.pt --out runs/workzone_depth/smoke --smoke
python artifacts/workzone_depth/run_experiment.py --moge path/to/model.pt --out runs/workzone_depth/full
```

准备脚本默认把切分写入 `runs/workzone_depth/prepared/data_splits`。已有切分时，用 `--splits` 指定包含 `workzone_depth.train.csv` 和 `workzone_depth.eval.csv` 的目录。距离头在训练集内部选轮次并用全部数值训练标签重训，MoGe 校准器用匹配到的训练样本拟合。固定的 250 张验证图用于最终比较。
