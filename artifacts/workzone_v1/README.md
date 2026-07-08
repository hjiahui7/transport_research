# workzone_v1

这是 work-zone-safety-rgbd-dataset 实验的轻量发布包，只包含可提交到 GitHub 的模型、CSV/JSON 结果和数据 split，不包含原始图片或训练过程目录。

## 文件结构

```text
artifacts/workzone_v1/
├─ models/
│  ├─ workzone_yolo11n_person_detector_best.pt
│  ├─ workzone_distance_head_base_yolo.pt
│  ├─ workzone_distance_head_ft_yolo.pt
│  └─ workzone_moge_calibrator.joblib
├─ qwen_prompt.md
├─ data_splits/
│  ├─ workzone_depth.train.csv
│  ├─ workzone_depth.eval.csv
│  └─ workzone_depth.all500.csv
└─ results/
   ├─ distance_eval_comparison.csv
   ├─ distance_eval_comparison.json
   ├─ scheme1_base_yolo_distance_eval_summary.json
   ├─ scheme1_finetuned_yolo_distance_eval_summary.json
   ├─ scheme2_moge_calibrated_eval_summary.json
   ├─ qwen3_vl_32b_eval20_pipeline_summary.json
   ├─ qwen3_vl_32b_eval20_pipeline_per_worker.csv
   ├─ qwen3_vl_32b_eval20_reports/
   ├─ qwen3_6_flash_all500_pipeline_summary.json
   ├─ qwen3_6_flash_all500_pipeline_per_worker.csv
   └─ qwen3_6_flash_all500_reports/
```

## 模型命名

| 文件 | 来源 | 用途 |
|---|---|---|
| `workzone_yolo11n_person_detector_best.pt` | `runs/yolo/workzone_yolo11n_960_e30/weights/best.pt` | work-zone fine-tuned YOLO 人体检测器 |
| `workzone_distance_head_base_yolo.pt` | `runs/workzone/workzone_yolo_distance_head.pt` | 原方案一 base YOLO distance head |
| `workzone_distance_head_ft_yolo.pt` | `runs/workzone/workzone_yolo_ft_distance_head.pt` | 新方案一 distance head |
| `workzone_moge_calibrator.joblib` | `runs/workzone/scheme2_workzone_calibrator.joblib` | 新方案二 MoGe 后处理 calibrator |

## 当前指标

| 方案 | Eval 口径 | 距离 MAE | RMSE | 0.5m 内 | 1.0m 内 | distance_band |
|---|---:|---:|---:|---:|---:|---:|
| 原方案一：base YOLO + distance head | 135 / 135 | 0.554m | 1.629m | 82.2% | 88.9% | 94.8% |
| 新方案一：fine-tuned YOLO + distance head | 135 / 135 | 0.428m | 1.088m | 84.4% | 93.3% | 91.1% |
| 新方案二：fine-tuned YOLO + MoGe + calibration | 134 / 135 | 0.379m | 0.983m | 89.6% | 95.5% | 94.8% |

20 图 JSON pipeline smoke test 结果在：

```text
results/qwen3_vl_32b_eval20_pipeline_summary.json
results/qwen3_vl_32b_eval20_pipeline_per_worker.csv
results/qwen3_vl_32b_eval20_reports/
```

500 图全量 JSON pipeline 结果在：

```text
results/qwen3_6_flash_all500_pipeline_summary.json
results/qwen3_6_flash_all500_pipeline_per_worker.csv
results/qwen3_6_flash_all500_reports/
```

| 字段 | 结果 |
|---|---:|
| 图片数 | `500` |
| worker 匹配 | `801 / 806 = 99.4%` |
| distance MAE | `0.188m` |
| distance_band | `626 / 656 = 95.4%` |
| high_visibility_vest | `774 / 786 = 98.5%` |
| helmet_status | `750 / 786 = 95.4%` |
| orientation | `642 / 787 = 81.6%` |
| occlusion_level | `601 / 793 = 75.8%` |
| VLM 设置 | `qwen3.6-flash, batch=10, workers=5` |

新增 500 图 VLM 对比结果：

```text
results/qwen3_vl_flash_all500_ft_head_summary.json
results/qwen3_vl_flash_all500_ft_head_per_worker.csv
results/qwen3_vl_flash_all500_ft_head_reports/
results/qwen2_5_vl_3b_local_all500_ft_head_summary.json
results/qwen2_5_vl_3b_local_all500_ft_head_per_worker.csv
results/qwen2_5_vl_3b_local_all500_ft_head_reports/
results/qwen3_6_flash_vlm_direct_distance_all500_summary.json
results/qwen3_6_flash_vlm_direct_distance_all500_per_worker.csv
results/qwen3_6_flash_vlm_direct_distance_all500_reports/
```

| 模型 / 路线 | 距离来源 | worker 匹配 | distance MAE | distance_band | high_visibility_vest | helmet_status | orientation | occlusion_level | 失败 batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.6-flash` baseline 属性-only | 本地 fine-tuned YOLO + distance head | `801 / 806 = 99.4%` | `0.188m` | `626 / 656 = 95.4%` | `774 / 786 = 98.5%` | `750 / 786 = 95.4%` | `642 / 787 = 81.6%` | `601 / 793 = 75.8%` | `0` |
| `qwen3-vl-flash` 属性-only | 本地 fine-tuned YOLO + distance head | `801 / 806 = 99.4%` | `0.188m` | `626 / 656 = 95.4%` | `692 / 786 = 88.0%` | `655 / 786 = 83.3%` | `486 / 787 = 61.8%` | `587 / 793 = 74.0%` | `0` |
| `Qwen2.5-VL-3B-Instruct` 本地属性-only | 本地 fine-tuned YOLO + distance head | `801 / 806 = 99.4%` | `0.188m` | `626 / 656 = 95.4%` | `721 / 786 = 91.7%` | `611 / 786 = 77.7%` | `276 / 787 = 35.1%` | `581 / 793 = 73.3%` | `3` |
| `qwen3.6-flash` VLM 直接估距 + 属性 | VLM 直接输出 distance_to_equipment_m | `801 / 806 = 99.4%` | `2.150m` | `397 / 656 = 60.5%` | `772 / 786 = 98.2%` | `751 / 786 = 95.5%` | `597 / 787 = 75.9%` | `593 / 793 = 74.8%` | `0` |

## 新方案一单图推理

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.workzone_report `
  --image work-zone-safety-rgbd-dataset\images\Garage1_000840.png `
  --checkpoint artifacts\workzone_v1\models\workzone_distance_head_ft_yolo.pt `
  --base-model artifacts\workzone_v1\models\workzone_yolo11n_person_detector_best.pt `
  --detector artifacts\workzone_v1\models\workzone_yolo11n_person_detector_best.pt `
  --out runs\workzone\single_report.json `
  --annotated-image runs\workzone\single_report_annotated.jpg `
  --device cuda:0
```

这条命令输出最终 JSON，其中距离来自本地 distance head，PPE/安全帽/朝向/遮挡来自 Qwen VLM。

## Qwen Prompt

Qwen VLM 的实际 prompt 已单独整理到：

```text
artifacts/workzone_v1/qwen_prompt.md
```

代码位置在：

```text
human_detect/workzone_report.py
```
