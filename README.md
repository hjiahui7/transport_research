# human_detect

这是一个轻量的人体检测 + 单图多人距离估计项目。当前重点不是先做大模型 3D 检测训练，而是把两条可跑通、可评估的方案整理清楚：

- 方案一：YOLO 特征 + 小 distance head，直接预测人体距离。
- 方案二：fine-tuned YOLO 检测人体，再用 MoGe 算深度，最后用 calibrator 修正距离。

## 环境

默认使用已有 conda 环境 `qwen`：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前机器测试过：

- GPU：RTX 4080 16GB
- PyTorch：`torch 2.5.1+cu121`
- 主要模型：
  - `YOLO11n-seg`：COCO person 实例分割
  - `YOLO11n / YOLO11s`：Rawalk person 检测训练
  - `Ruicheng/moge-2-vits-normal`：metric depth 和相机 FOV/内参估计

运行测试：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m pytest -q
```

## 数据目录

PM-HMCW 数据默认放在：

```text
data/pm_hmcw/raw/real-world/test/{image_2,label_2,calib}
data/pm_hmcw/raw/virtual/test/{image_2,label_2,calib}
```

Rawalk / EgoHumans 数据默认放在：

```text
data/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging
```

`data/` 和 `runs/` 已经被 `.gitignore` 忽略，不应该提交数据和模型权重。

## 当前 Rawalk 实验总结

距离 GT 不是模型推理出来的。Rawalk/EgoHumans 存了 3D 人体关键点和每帧 Aria 相机标定，所以这里用它们计算：

```text
fit_poses3d 3D keypoints
-> ego Aria world-to-camera calibration
-> 投影出 2D bbox
-> depth_m = 相机坐标系 z 深度
-> distance_m = sqrt(x^2 + y^2 + z^2)
```

### GT 是怎么计算的

Rawalk ego 距离 GT 来自数据集自带的 3D 信息，主要读取这些文件：

```text
processed_data/fit_poses3d/*.npy
ego/ariaXX/calib/*.txt
ego/ariaXX/images/rgb/*.jpg
colmap/workplace/colmap_from_aria_transforms.pkl
```

计算流程：

1. 读取每一帧每个人的 3D 人体关键点。
2. 读取对应 ego Aria 相机的内参、外参和 Aria 到统一世界坐标的变换。
3. 把人体 3D 点从世界坐标系变换到当前相机坐标系。
4. 把相机坐标系下的 3D 点投影到图片上，得到这个人的 2D bbox。
5. 用相机坐标系下的点计算距离：

```text
depth_m = Z
distance_m = sqrt(X^2 + Y^2 + Z^2)
```

这里 `depth_m` 是相机正前方方向的深度，`distance_m` 是相机中心到人的真实直线距离。当前实现会对可见关键点求平均，并记录 `torso_depth_m` / `torso_distance_m` 作为额外参考字段。

生成 GT 的入口：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.rawalk_ego_depth `
  --rawalk-root data\media\rawalk `
  --viewers aria01 aria02 aria03 aria04 `
  --frame-step 10 `
  --out runs\rawalk_ego_depth_all_step10.csv
```

### 使用的数据

| 数据文件/目录 | 来源 | 图片数 | 人体标签/行数 | 用途 |
|---|---|---:|---:|---|
| `data/rawalk_yolo_person` train | Rawalk exo bbox 标注 | 3416 | 11280 个 bbox | 训练 YOLO 人体检测器 |
| `data/rawalk_yolo_person` val | Rawalk exo bbox 标注 | 296 | 975 个 bbox | 验证 YOLO 人体检测器 |
| `runs/rawalk_ego_depth_all_step10.csv` | Rawalk ego 3D 关键点 + 相机标定 | 1880 | 3924 个距离 GT | 过滤前的 ego 距离标签 |
| `runs/rawalk_ego_depth_all_step10_m20.train.csv` | 过滤后的 ego GT，`0.2m <= distance_m <= 20m` | 1449 | 2990 个距离 GT | 方案一 distance head 训练；方案二 calibration 训练来源 |
| `runs/rawalk_ego_depth_all_step10_m20.eval.csv` | 同一个固定切分 | 362 | 728 个距离 GT | 两个方案共用的 eval set |

`max-distance 20m` 过滤掉了 206 个明显异常的投影/标定样本。train/eval 按 `image_path` 分组切分，所以同一张图里的人不会一部分进 train、一部分进 eval。

### 两个方案的结果

| 方案 | 检测/距离来源 | 训练数据 | Eval set | 匹配/评估人数 | 距离 MAE | 0.5m 内 | 1.0m 内 | Bias |
|---|---|---|---|---:|---:|---:|---:|---:|
| 方案一：YOLO distance head | 冻结官方 `yolo11n.pt` 特征，只训练小的 YOLO-grid distance head | 2990 个 ego GT 人体距离标签 | 同一个 728 人 eval CSV | 728 / 728 | **0.308m** | **82.6%** | **94.5%** | -0.051m |
| 方案二 raw：fine-tuned YOLO + MoGe | `rawalk_yolo11s_960_e20` 检测器 + MoGe 深度几何计算 | 无距离校准 | 同一个 728 人 eval CSV | 519 / 728 | 2.596m | 0.4% | 2.3% | +2.588m |
| 方案二 calibrated：fine-tuned YOLO + MoGe + calibrator | 同一个检测器和深度，再接 MLP calibration | 2015 个 train 匹配样本 | 同一个 728 人 eval CSV | 519 / 728 | **0.220m** | **90.8%** | **98.3%** | -0.022m |

结果解释：

- 方案一是在 GT 框中心格子上评估 distance head，衡量的是距离头本身的能力，还没有把检测漏检算进去。
- 方案二跑的是完整链路：检测 -> MoGe 深度 -> calibration。匹配上的人距离更准，但当前检测/匹配只覆盖 `519 / 728 = 71.3%` 的 eval GT 人体。
- 如果把漏检也算失败，方案二 calibrated 在全部 728 个 GT 上，0.5m 内比例是 `64.7%`，1.0m 内比例是 `70.1%`。

主要输出文件：

```text
runs/yolo/rawalk_yolo11s_960_e20/weights/best.pt
runs/yolo_distance_head_all_step10_m20.pt
runs/yolo_distance_head_all_step10_m20.metrics.json
runs/rawalk_ego_scheme2_train_preds.csv
runs/rawalk_ego_scheme2_eval_preds.csv
runs/rawalk_ego_scheme2_calibrator.joblib
runs/rawalk_ego_scheme2_eval_summary.json
```

为了方便 GitHub 同步，当前把处理后的 CSV/JSON/YAML 结果复制了一份到：

```text
artifacts/data/
```

`artifacts/data/` 包含 train/eval CSV、预测 CSV、metrics 和 summary。它不包含 Rawalk 原始图片、原始 3D pose 和模型权重，因为完整 `data/` 当前约 84GB，模型权重也作为二进制产物单独管理。

## 单图推理

带 PM-HMCW 标定文件时，会用 KITTI `P2` 内参覆盖 MoGe 估计内参：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image data\pm_hmcw\raw\real-world\test\image_2\000248.png `
  --calib data\pm_hmcw\raw\real-world\test\calib\000248.txt `
  --out runs\smoke_000248.json `
  --vis runs\smoke_000248.png `
  --imgsz 640 `
  --geom-size 640 `
  --num-tokens 1200 `
  --device cuda:0 `
  --half
```

没有标定文件时，会用 MoGe 估计出来的内参：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image data\pm_hmcw\raw\real-world\test\image_2\000248.png `
  --out runs\smoke_000248_no_calib.json `
  --vis runs\smoke_000248_no_calib.png
```

## MoGe 在这里做什么

MoGe 是一个单目 3D 几何估计模型。单目意思是只输入一张普通 RGB 图片，不需要双目相机或 LiDAR。

我们使用的是：

```text
Ruicheng/moge-2-vits-normal
```

在这个项目里，MoGe 主要输出两类东西：

```text
depth map: 每个像素的 metric depth
camera intrinsics / FOV: 没有标定文件时估计相机内参
```

方案二里它的作用是：

```text
图片
-> YOLO 找到人在哪里
-> MoGe 给整张图估计 depth map
-> 从人的 bbox/mask 区域取深度中位数
-> 用相机内参反投影成 3D 点
-> distance_m = sqrt(X^2 + Y^2 + Z^2)
```

大白话：YOLO 回答“人在哪里”，MoGe 回答“图里每个地方大概离相机多远”。MoGe 不负责识别人，也不是人体检测模型。

当前代码优先取人区域下方 60% 的深度中位数。这样做是为了减少头部、背景边缘、框漏背景造成的深度噪声。没有足够像素时，会退回到全 bbox/mask 的深度中位数。

MoGe 的 raw depth 在 Rawalk ego 上有明显系统偏差：eval matched raw distance MAE 是 `2.596m`，平均偏远 `+2.588m`。所以方案二后面加了 calibrator，把 matched-person distance MAE 降到 `0.220m`。

## 方案一：YOLO Distance Head

这条路线不使用 MoGe，也不使用 calibration。流程是：

```text
图片 -> 冻结 YOLO backbone/neck 特征 -> 小 YOLO-grid distance head -> distance_m
```

当前训练使用：

- base model：官方 `yolo11n.pt`
- YOLO 主体：冻结，不训练
- 新增 head：`YoloGridDistanceHead`
- 训练数据：`runs/rawalk_ego_depth_all_step10_m20.train.csv`
- eval 数据：`runs/rawalk_ego_depth_all_step10_m20.eval.csv`
- epoch：30
- batch：16
- image size：640

训练命令：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.train_yolo_distance_head `
  --labels runs\rawalk_ego_depth_all_step10_m20.train.csv `
  --val-labels runs\rawalk_ego_depth_all_step10_m20.eval.csv `
  --model yolo11n.pt `
  --out runs\yolo_distance_head_all_step10_m20.pt `
  --metrics-out runs\yolo_distance_head_all_step10_m20.metrics.json `
  --epochs 30 `
  --batch 16 `
  --imgsz 640 `
  --device cuda:0
```

当前 eval 结果：

```text
eval images: 362
eval persons: 728
MAE: 0.308m
RMSE: 0.486m
Bias: -0.051m
Median absolute error: 0.192m
<= 0.5m: 82.6%
<= 1.0m: 94.5%
```

注意：这个评估是“用 GT bbox 中心格子读 distance head”，还没有经过完整检测链路。

## 方案二：Fine-tuned YOLO + MoGe + Calibration

这条路线不让 YOLO 直接预测距离。YOLO 只负责检测人体，MoGe 负责估计深度，calibrator 负责修正系统误差。

流程：

```text
图片
-> fine-tuned YOLO 检测人体 bbox
-> MoGe 估计整图 depth map 和相机内参/FOV
-> bbox/mask 区域取深度中位数
-> 用内参反投影成 3D 点并计算 distance_m
-> calibrator 修正距离
```

### Fine-tuned YOLO 检测器

Rawalk 当前提供 bbox，不提供 segmentation mask，所以先训练 detect-only YOLO person detector。

准备 YOLO 数据：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.prepare_rawalk_yolo `
  --rawalk-root data\media\rawalk `
  --out data\rawalk_yolo_person `
  --streams exo `
  --frame-step 10 `
  --val-fraction 0.2
```

训练当前使用的 YOLO11s 检测器：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.train_yolo `
  --data data\rawalk_yolo_person\rawalk_person.yaml `
  --model yolo11s.pt `
  --epochs 20 `
  --imgsz 960 `
  --batch 8 `
  --device cuda:0
```

检测器结果：

```text
model: runs/yolo/rawalk_yolo11s_960_e20/weights/best.pt
precision: 0.956
recall: 0.873
mAP50: 0.933
mAP50-95: 0.854
```

### Rawalk Ego 距离 GT

生成 ego 距离标签：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.rawalk_ego_depth `
  --rawalk-root data\media\rawalk `
  --viewers aria01 aria02 aria03 aria04 `
  --frame-step 10 `
  --out runs\rawalk_ego_depth_all_step10.csv
```

固定切分，并过滤明显异常距离：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.split_rawalk_ego_depth `
  --labels runs\rawalk_ego_depth_all_step10.csv `
  --train-out runs\rawalk_ego_depth_all_step10_m20.train.csv `
  --eval-out runs\rawalk_ego_depth_all_step10_m20.eval.csv `
  --summary-out runs\rawalk_ego_depth_all_step10_m20.split.json `
  --eval-fraction 0.2 `
  --seed 7 `
  --min-distance 0.2 `
  --max-distance 20
```

### 生成 MoGe 预测并训练 Calibrator

先在 train split 上跑检测 + MoGe，并和 GT bbox 匹配：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.eval_rawalk_ego_depth `
  --labels runs\rawalk_ego_depth_all_step10_m20.train.csv `
  --out runs\rawalk_ego_scheme2_train_preds.csv `
  --detector runs\yolo\rawalk_yolo11s_960_e20\weights\best.pt `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0 `
  --max-distance 20 `
  --iou-threshold 0.3
```

再在 eval split 上跑同样链路：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.eval_rawalk_ego_depth `
  --labels runs\rawalk_ego_depth_all_step10_m20.eval.csv `
  --out runs\rawalk_ego_scheme2_eval_preds.csv `
  --detector runs\yolo\rawalk_yolo11s_960_e20\weights\best.pt `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0 `
  --max-distance 20 `
  --iou-threshold 0.3
```

只用 train 匹配结果训练 calibrator：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.fit_calibrator `
  --preds runs\rawalk_ego_scheme2_train_preds.csv `
  --out runs\rawalk_ego_scheme2_calibrator.joblib `
  --metrics-out runs\rawalk_ego_scheme2_calibrator.metrics.json `
  --model best `
  --group-column image_id
```

### Calibration 是怎么做的

Calibration 不是重新训练 YOLO，也不是重新训练 MoGe。它是一个很小的后处理回归器，学习“MoGe 算出来的距离应该怎么修正到 GT 距离”。

训练数据来自：

```text
runs/rawalk_ego_scheme2_train_preds.csv
```

这个 CSV 是这样生成的：

```text
train 图片
-> fine-tuned YOLO 检测人框
-> MoGe 估计 depth map 和 FOV/内参
-> 对每个检测框算 raw z_depth_m / raw distance_m
-> 和 Rawalk GT bbox 做 IoU 匹配
-> 匹配成功后得到一行：预测特征 + GT 距离
```

每一行训练样本大概长这样：

```text
输入特征:
  raw z_depth_m
  raw distance_m
  bbox 宽高、面积、中心位置
  mask/bbox 区域面积
  detection score
  yaw / pitch
  FOV
  depth p10/p25/p50/p75/p90
  lower-depth p10/p25/p50/p75/p90

监督目标:
  z_gt
  distance_gt
```

这些输入特征在新图推理时也能拿到，不是人工标注：

| 输入特征 | 含义 |
|---|---|
| `z_depth_m` | MoGe 算出来的原始前向深度，也就是相机坐标系 Z |
| `distance_m` | 用 MoGe 深度和相机内参算出来的原始直线距离 |
| `bbox_width_norm` | 人框宽度 / 图片宽度 |
| `bbox_height_norm` | 人框高度 / 图片高度 |
| `bbox_area_norm` | 人框面积 / 图片面积 |
| `mask_area_norm` | mask 或 bbox 区域面积 / 图片面积 |
| `center_x_norm` | 人框中心 x 坐标 / 图片宽度 |
| `center_y_norm` | 人框中心 y 坐标 / 图片高度 |
| `score` | YOLO 检测置信度 |
| `bearing_yaw_deg` | 人相对相机的左右角度 |
| `elevation_pitch_deg` | 人相对相机的上下角度 |
| `fov_deg` | 相机水平视场角；没有标定时来自 MoGe 估计 |
| `depth_p10_m` / `p25` / `p50` / `p75` / `p90` | 人框/mask 区域内 MoGe depth 的分位数 |
| `lower_depth_p10_m` / `p25` / `p50` / `p75` / `p90` | 人框/mask 下方 60% 区域内 MoGe depth 的分位数 |

`lower_depth_*` 不是额外标签。它们是现场从 MoGe depth map 里算出来的：YOLO 给人框或 mask，MoGe 给每个像素的 depth，然后只取人区域下方 60% 的 depth 分布做统计。如果检测器是 detect-only 没有 mask，就用 bbox 当粗 mask。

训练时的监督目标是：

| 输出/目标 | 含义 |
|---|---|
| `z_gt` | Rawalk 3D GT 算出来的真实前向深度 |
| `distance_gt` | Rawalk 3D GT 算出来的真实相机到人的直线距离 |

保存的 calibrator 里实际有两个回归器：

| 回归器 | 输入 | 输出 |
|---|---|---|
| `z_model` | 上面的全部输入特征 | `z_depth_calibrated_m` |
| `distance_model` | 上面的全部输入特征 | `distance_calibrated_m` |

`fit_calibrator.py` 会尝试几种小模型：

```text
scale_bias
linear
ridge
gradient boosting regressor
MLP regressor
```

然后在 train 内部按 `image_id` 再切一个 held-out 验证集，选择 distance MAE 最低的模型。当前 Rawalk scheme-2 选择的是 `MLP`。选好模型类型后，再用全部 train 匹配样本重新拟合并保存：

```text
runs/rawalk_ego_scheme2_calibrator.joblib
```

关键点：

- eval CSV 不参与 calibrator 训练。
- calibration 只修正 `z_depth_m` 和 `distance_m`，不改变检测框。
- 如果检测器漏人，calibrator 无法补回来；所以方案二要同时看 `match rate` 和 matched-person distance MAE。

方案二最终 eval 结果保存在：

```text
runs/rawalk_ego_scheme2_eval_summary.json
```

关键结果：

```text
eval GT persons: 728
matched persons: 519
match rate: 71.3%

raw MoGe distance MAE: 2.596m
calibrated distance MAE: 0.220m
calibrated <= 0.5m on matched persons: 90.8%
calibrated <= 1.0m on matched persons: 98.3%
calibrated bias: -0.022m
```

## 代码入口

常用模块：

```text
human_detect.infer                     单图推理 CLI
human_detect.rawalk_ego_depth          从 Rawalk 3D pose + calib 生成 ego 距离 GT
human_detect.split_rawalk_ego_depth    固定切分 train/eval，并过滤异常距离
human_detect.prepare_rawalk_yolo       准备 Rawalk YOLO bbox 数据
human_detect.train_yolo                训练 YOLO person detector
human_detect.train_yolo_distance_head  训练方案一 distance head
human_detect.eval_rawalk_ego_depth     跑方案二检测+深度并输出匹配预测
human_detect.fit_calibrator            训练轻量 calibration regressor
```
