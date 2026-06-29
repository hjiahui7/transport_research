# rawalk_v1

这是 Rawalk/EgoHumans 旧实验的轻量模型发布目录。根目录 `models/` 里仍保留原始文件名用于兼容旧命令；这里使用更清楚的文件名，方便区分方案一和方案二。

## 模型文件

| 文件 | 原始来源 | 用途 |
|---|---|---|
| `models/rawalk_scheme1_yolo11n_base.pt` | `models/yolo11n.pt` | 方案一 distance head 使用的 YOLO11n base |
| `models/rawalk_scheme1_distance_head.pt` | `models/yolo_distance_head_all_step10_m20.pt` | 方案一 Rawalk distance head |
| `models/rawalk_scheme2_yolo11s_person_detector_best.pt` | `models/rawalk_yolo11s_960_e20_best.pt` | 方案二 Rawalk fine-tuned YOLO person detector |
| `models/rawalk_scheme2_moge_calibrator.joblib` | `models/rawalk_ego_scheme2_calibrator.joblib` | 方案二 MoGe 距离校准器 |

## 对应结果

Rawalk/EgoHumans 的 CSV、JSON、YAML 结果仍在：

```text
artifacts/data/
```

主要指标见项目根目录 README 的“Rawalk/EgoHumans 旧数据”小节。
