# Work-zone v2 实验产物

本目录对应 `500 + 1185 = 1685` 张 RGB 工地图像的新实验。人体检测和距离 calibration 已重新训练；最终 250 图完整链路使用 `qwen3.6-flash` 重新判断安全属性。

## 目录结构

```text
workzone_v2/
├─ README.md
├─ qwen_prompt.md
├─ data_splits/
│  ├─ workzone_depth.train.csv
│  ├─ workzone_depth.eval.csv
│  └─ workzone_depth.all1685.csv
├─ models/
│  ├─ workzone_yolo11n_person_detector_best.pt
│  ├─ workzone_distance_head_ft_yolo.pt
│  └─ workzone_moge_calibrator.joblib
└─ results/
   ├─ workzone_prepare_summary.json
   ├─ yolo11n_baseline_metrics.json
   ├─ workzone_yolo11n_detector_metrics.json
   ├─ workzone_distance_head_ft_metrics.json
   ├─ scheme1_finetuned_yolo_distance_eval_summary.json
   ├─ workzone_moge_calibrator_metrics.json
   ├─ scheme2_moge_raw_eval_summary.json
   ├─ scheme2_moge_calibrated_eval_summary.json
   ├─ distance_eval_comparison.csv
   ├─ vlm_eval250_comparison.csv
   ├─ qwen3_6_flash_eval250_moge_calibrated_summary.json
   ├─ qwen3_6_flash_eval250_moge_calibrated_per_worker.csv
   └─ qwen3_6_flash_eval250_moge_calibrated_reports/
```

## 数据与切分

| 项目 | 图像 | worker | 有数值距离 GT 的 worker |
|---|---:|---:|---:|
| 全部 | 1685 | 3278 | 3013 |
| Train | 1435 | 2863 | 2674 |
| Validation | 250 | 415 | 339 |

- Wave 1：500 张图，包含 bbox、LiDAR 距离和 VLM 安全属性 GT。
- Wave 2：1185 张图，包含 2472 个 bbox，其中 2356 个有数值 LiDAR 距离；不包含 vest、helmet、orientation、occlusion GT。
- Validation 固定从 Wave 1 六个 recording 中按比例抽取 250 张，随机种子为 `7`。
- Train 使用 Wave 1 剩余 250 张和 Wave 2 全部 1185 张。
- 距离区间统一为 `<3m`、`3-6m`、`>6m`。

## YOLO 人体检测

v2 从官方 `yolo11n.pt` 重新训练，未继承 v1 权重，避免 v1 见过新 validation 图片。训练配置为 30 epoch、`imgsz=960`、batch 8、FP16、冻结前 10 层。

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 官方 YOLO11n baseline | 49.6% | 87.7% | 64.6% | 59.1% |
| v2 fine-tuned YOLO11n | **98.0%** | **96.4%** | **98.1%** | **88.9%** |

## 三种距离方案

`fine-tuned YOLO + distance head` 也使用 v2 新数据重新训练：冻结 v2 YOLO 主干，在 2674 个 train 距离 GT 上训练轻量 head 30 epoch，最佳为第 7 轮。下面三行全部使用同一批 250 图 validation。

| 距离方案 | 成功匹配 / 数值 GT | MAE | RMSE | 0.5m 内 | 1.0m 内 | 档位准确率 |
|---|---:|---:|---:|---:|---:|---:|
| fine-tuned YOLO + distance head | 332 / 339 | 0.371m | 1.145m | 88.9% | 96.4% | **93.1%** |
| fine-tuned YOLO + 原始 MoGe | 332 / 339 | 1.856m | 3.454m | 12.0% | 32.8% | 69.9% |
| fine-tuned YOLO + MoGe + MLP calibration | 332 / 339 | **0.334m** | **1.089m** | **92.8%** | **96.7%** | 92.5% |

Distance head 的推理更轻，不需要运行 MoGe；MoGe + calibration 的 MAE 和 0.5m 内命中率更好。

## MoGe 与 Calibration

MoGe 冻结，不训练模型参数。它对整图生成 metric depth 和相机 FOV；程序在每个 worker 区域汇聚深度，再将 bbox、mask、深度分位数、画面位置、置信度和 FOV 等特征输入小型 calibrator。

Calibrator 只使用 train 的 2672 条成功匹配记录训练，在 train 内按 image_id 留出 20% 比较 scale/bias、linear、ridge、GBR 和 MLP，最终选择 MLP。固定 validation 不参与 calibrator 拟合。

这里的 `332 / 339` 是成功匹配 worker 数 / 有数值距离 GT 的 worker 数，不是图片数。Validation 始终是 250 张图。

## VLM 属性结果

以下前三行从 v1 已完成的 all500 结果中筛选相同 250 图重新计算，API 调用数为 0；最后一行是 v2 重新调用的 `qwen3.6-flash`。为了只比较 VLM 属性能力，表中所有模型统一接 v2 `MoGe + calibration` 距离模块，因此距离 MAE 统一写为 `0.334m`，距离档位统一为 `92.5%`。历史 summary JSON 中的原始距离结果没有修改。

| 方案 | worker 匹配 | 距离 MAE | 距离档位 | Vest | Helmet | Orientation | Occlusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史 qwen3.6-flash（离线筛选） | 407 / 415 | 0.334m | 92.5% | 97.8% | 95.3% | 84.6% | 75.4% |
| 历史 qwen3-vl-flash（离线筛选） | 407 / 415 | 0.334m | 92.5% | 86.6% | 80.9% | 62.0% | 73.6% |
| 历史 Qwen2.5-VL-3B（离线筛选） | 407 / 415 | 0.334m | 92.5% | 91.1% | 76.0% | 34.2% | 72.7% |
| **v2 qwen3.6-flash + MoGe calibration** | **407 / 415** | **0.334m** | **92.5%** | **98.0%** | **94.2%** | **83.5%** | **76.4%** |

VLM 只判断 `high_visibility_vest`、`helmet_status`、`orientation` 和 `occlusion_level`。`distance_to_equipment_m` 来自本地 YOLO + MoGe + calibration，`distance_band` 由代码按米数计算，不让 VLM 猜距离。

## 最终 JSON 流程

```text
输入单张 RGB 图
  -> fine-tuned YOLO 检测 worker 和 bbox
  -> MoGe 生成整图深度与 FOV
  -> MLP calibration 修正每个 worker 的米数
  -> 代码计算 Close / Careful / Safe
  -> 带 W1/W2 编号框的图片交给 qwen3.6-flash
  -> 合并距离与安全属性，输出单图 JSON
```

最终 250 份 JSON 在 `results/qwen3_6_flash_eval250_moge_calibrated_reports/`。VLM prompt 在 `qwen_prompt.md`。

## 复现命令

准备数据：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.prepare_workzone_v2
```

训练 YOLO：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.train_yolo `
  --data data\workzone_v2_yolo_person\workzone_person.yaml `
  --model yolo11n.pt --epochs 30 --imgsz 960 --batch 8 `
  --device cuda:0 --freeze 10 --patience 8 --amp
```

调用 VLM 前只在本地环境设置 `QWEN_API_KEY`，不要把真实 Key 写入 README、代码或 Git。
