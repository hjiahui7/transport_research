# human_detect

这是一个人体检测 + 单图多人距离估计项目。当前只围绕 Rawalk/EgoHumans 数据整理两条路线：

- **方案一：YOLO distance head**  
  YOLO 找人框，然后一个小的 distance head 直接预测人到相机的距离。

- **方案二：fine-tuned YOLO + MoGe + calibration**  
  fine-tuned YOLO 负责找人，MoGe 负责估深度，calibrator 负责把 MoGe 的系统误差修正到 Rawalk GT。

## 当前结果

两个方案使用同一个 eval set：

```text
runs/rawalk_ego_depth_all_step10_m20.eval.csv
362 张图，728 个 GT 人
```

| 方案 | 检测/距离来源 | 训练数据 | Eval 口径 | 距离 MAE | 0.5m 内 | 1.0m 内 | Bias |
|---|---|---|---:|---:|---:|---:|---:|
| 方案一：YOLO distance head | 官方 `yolo11n.pt` 特征 + 小 distance head | 2990 个 ego GT 距离标签 | 728 / 728 | **0.308m** | **82.6%** | **94.5%** | -0.051m |
| 方案二 raw：fine-tuned YOLO + MoGe | `rawalk_yolo11s_960_e20` + MoGe depth | 无 calibration | 519 / 728 | 2.596m | 0.4% | 2.3% | +2.588m |
| 方案二 calibrated：fine-tuned YOLO + MoGe + MLP calibrator | 同上，再接 calibration | 2015 个 train 匹配样本 | 519 / 728 | **0.220m** | **90.8%** | **98.3%** | -0.022m |

解释：

- 方案一目前是在 GT 框中心格子上评估 distance head，主要衡量距离头本身，还没把检测漏检算进去。
- 方案二是完整链路评估：检测 -> MoGe -> calibration -> 和 GT 匹配。它匹配上的人距离更准，但检测/匹配覆盖率是 `519 / 728 = 71.3%`。
- 如果把漏检也算失败，方案二 calibrated 在全部 728 个 GT 上，0.5m 内比例是 `64.7%`，1.0m 内比例是 `70.1%`。

## 环境

默认使用已有 conda 环境 `qwen`：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前测试环境：

```text
GPU: RTX 4080 16GB
PyTorch: torch 2.5.1+cu121
Python: 3.10
```

运行测试：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m pytest -q
```

## 快速推理

如果你有本地包：

```text
human_detect_models_data_bundle.zip
```

把它解压到项目根目录后，会得到：

```text
models/
artifacts/data/
README_BUNDLE.md
```

这个包包含小模型和结果 CSV/JSON/YAML，不包含 Rawalk 原始大数据。

### 方案一推理

方案一不跑 MoGe，不跑 calibration，直接用 YOLO box + distance head 输出距离。

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer_distance_head `
  --image path\to\image.jpg `
  --checkpoint models\yolo_distance_head_all_step10_m20.pt `
  --base-model models\yolo11n.pt `
  --out runs\scheme1_distance_head.json `
  --vis runs\scheme1_distance_head.png `
  --device cuda:0 `
  --imgsz 640
```

输出里看：

```text
persons[].distance_m
```

### 方案二推理

方案二会跑 MoGe，所以环境里需要能加载 `Ruicheng/moge-2-vits-normal`，或者本机 Hugging Face 缓存里已有模型。

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image path\to\image.jpg `
  --detector models\rawalk_yolo11s_960_e20_best.pt `
  --calibrator models\rawalk_ego_scheme2_calibrator.joblib `
  --out runs\scheme2_moge_calibrated.json `
  --vis runs\scheme2_moge_calibrated.png `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0
```

输出里看：

```text
persons[].distance_m              # MoGe raw distance
persons[].distance_calibrated_m   # calibration 后距离
```

## 数据和产物

Rawalk / EgoHumans 原始数据默认放在：

```text
data/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging
```

GitHub 里放轻量模型和结果文件：

```text
models/
artifacts/data/
```

本地大数据不进 GitHub：

```text
data/media                  # 原始 Rawalk，大约 74GB
data/rawalk_yolo_person     # YOLO 训练图片和标签，大约 3.36GB
runs/                       # 训练过程和模型输出
```

`models/` 里包含直接推理需要的小模型：

| 文件 | 用途 |
|---|---|
| `models/yolo11n.pt` | 方案一 distance head 的 YOLO base model |
| `models/yolo_distance_head_all_step10_m20.pt` | 方案一训练好的 distance head |
| `models/rawalk_yolo11s_960_e20_best.pt` | 方案二 fine-tuned Rawalk YOLO detector |
| `models/rawalk_ego_scheme2_calibrator.joblib` | 方案二 MLP calibrator |

`artifacts/data/` 里主要是：

| 文件 | 用途 |
|---|---|
| `rawalk_ego_depth_all_step10.csv` | 过滤前 ego 距离 GT |
| `rawalk_ego_depth_all_step10_m20.train.csv` | 过滤后的 train GT |
| `rawalk_ego_depth_all_step10_m20.eval.csv` | 过滤后的 eval GT |
| `rawalk_ego_scheme2_train_preds.csv` | 方案二 train 匹配结果，用来训练 calibrator |
| `rawalk_ego_scheme2_eval_preds.csv` | 方案二 eval 匹配结果 |
| `rawalk_ego_scheme2_eval_summary.json` | 方案二 eval 指标 |
| `yolo_distance_head_all_step10_m20.metrics.json` | 方案一训练指标 |
| `rawalk_yolo11s_960_e20.metrics.json` | YOLO 检测器指标 |

## 从原始 Rawalk 处理到训练数据

别人下载 Rawalk/EgoHumans 原始数据后，可以按下面两条路线复现。两条路线共用同一份 Rawalk 原始数据，但训练目标不同：

- 方案一：只训练一个 YOLO distance head，输入单图后直接给人框和距离。
- 方案二：先训练 Rawalk YOLO 检测器，再用 MoGe 算深度，最后用 calibrator 修正距离。

### 需要的原始子目录

代码会从 `data/media/rawalk` 自动解析到最终的 `01_tagging` 目录。实际用到：

```text
01_tagging/{sequence}/exo/{cam}/images/*.jpg
01_tagging/{sequence}/processed_data/bboxes/{cam}/rgb/*.npy
01_tagging/{sequence}/ego/{aria}/images/rgb/*.jpg
01_tagging/{sequence}/ego/{aria}/calib/*.txt
01_tagging/{sequence}/processed_data/fit_poses3d/*.npy
01_tagging/{sequence}/colmap/workplace/colmap_from_aria_transforms.pkl
```

用途：

| 原始数据 | 用途 |
|---|---|
| `exo/*/images` | YOLO 检测器训练图片 |
| `processed_data/bboxes/*/rgb` | YOLO 检测器 bbox 标签 |
| `ego/*/images/rgb` | ego 图片；方案二还会拿它做检测 + MoGe 深度输入 |
| `ego/*/calib` | ego 相机内参/外参，用于 GT 投影和距离计算 |
| `processed_data/fit_poses3d` | 3D 人体关键点，用于生成距离 GT |
| `colmap_from_aria_transforms.pkl` | 多个 Aria 相机到统一世界坐标的变换 |

### 方案一：Distance Head 路线

这条路线不跑 MoGe，也不训练 Rawalk YOLO 检测器。它用 Rawalk ego 的 3D pose/calib 先生成距离 GT，然后在 `yolo11n.pt` 上接一个很小的 distance head。

#### 1. 生成 Rawalk ego 距离 GT

这一步不训练模型，只用 Rawalk 的 3D pose + Aria calibration 生成 `bbox + depth_m + distance_m` CSV。

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.rawalk_ego_depth `
  --rawalk-root data\media\rawalk `
  --viewers aria01 aria02 aria03 aria04 `
  --frame-step 10 `
  --out runs\rawalk_ego_depth_all_step10.csv
```

当前生成：

```text
images: 1880
person distance rows: 3924
```

#### 2. 过滤异常距离并固定 train/eval

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

当前 split：

```text
original rows: 3924
filtered outliers: 206
train images: 1449
train person rows: 2990
eval images: 362
eval person rows: 728
```

切分按 `image_path` 分组，避免同一张图里的人同时出现在 train 和 eval。

#### 3. 训练方案一 Distance Head

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

当前训练出的模型已复制到：

```text
models/yolo_distance_head_all_step10_m20.pt
```

#### 4. 方案一单图推理

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer_distance_head `
  --image path\to\image.jpg `
  --checkpoint models\yolo_distance_head_all_step10_m20.pt `
  --base-model models\yolo11n.pt `
  --out runs\scheme1_distance_head.json `
  --vis runs\scheme1_distance_head.png `
  --device cuda:0 `
  --imgsz 640
```

方案一输出的是模型自己预测的 `distance_m`，不依赖 MoGe depth，也不做 calibration。

### 方案二：YOLO + MoGe + Calibration 路线

这条路线是完整链路：先提高人体检测，再用 MoGe 从单图估计深度，最后用 Rawalk GT 训练一个小 calibrator 修正 MoGe 的系统偏差。

#### 1. 生成 YOLO 训练数据

把 Rawalk exo 图片和 bbox 标注转成 YOLO 格式：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.prepare_rawalk_yolo `
  --rawalk-root data\media\rawalk `
  --out data\rawalk_yolo_person `
  --streams exo `
  --frame-step 10 `
  --val-fraction 0.2
```

输出：

```text
data/rawalk_yolo_person/
├─ images/train
├─ images/val
├─ labels/train
├─ labels/val
└─ rawalk_person.yaml
```

当前生成数量：

```text
train images: 3416
train boxes: 11280
val images: 296
val boxes: 975
```

#### 2. 训练 Rawalk YOLO 检测器

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.train_yolo `
  --data data\rawalk_yolo_person\rawalk_person.yaml `
  --model yolo11s.pt `
  --epochs 20 `
  --imgsz 960 `
  --batch 8 `
  --device cuda:0
```

当前使用的检测器：

```text
runs/yolo/rawalk_yolo11s_960_e20/weights/best.pt
```

检测指标：

```text
precision: 0.956
recall: 0.873
mAP50: 0.933
mAP50-95: 0.854
```

当前训练出的检测器已复制到：

```text
models/rawalk_yolo11s_960_e20_best.pt
```

#### 3. 生成/复用 ego 距离 GT

方案二使用和方案一相同的 ego GT。如果已经跑过方案一的第 1、2 步，直接复用下面两个文件：

```text
runs/rawalk_ego_depth_all_step10_m20.train.csv
runs/rawalk_ego_depth_all_step10_m20.eval.csv
```

如果只跑方案二，也可以在本块里重新生成：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.rawalk_ego_depth `
  --rawalk-root data\media\rawalk `
  --viewers aria01 aria02 aria03 aria04 `
  --frame-step 10 `
  --out runs\rawalk_ego_depth_all_step10.csv
```

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

#### 4. 生成方案二 Calibration 训练数据

train split：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.eval_rawalk_ego_depth `
  --labels runs\rawalk_ego_depth_all_step10_m20.train.csv `
  --out runs\rawalk_ego_scheme2_train_preds.csv `
  --detector models\rawalk_yolo11s_960_e20_best.pt `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0 `
  --max-distance 20 `
  --iou-threshold 0.3
```

eval split：

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.eval_rawalk_ego_depth `
  --labels runs\rawalk_ego_depth_all_step10_m20.eval.csv `
  --out runs\rawalk_ego_scheme2_eval_preds.csv `
  --detector models\rawalk_yolo11s_960_e20_best.pt `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0 `
  --max-distance 20 `
  --iou-threshold 0.3
```

这一步会对每张 ego 图跑 `YOLO -> MoGe -> 几何反投影`，再和 Rawalk GT 按 IoU 匹配，输出每个人的 raw 距离、深度统计特征和 GT 距离。

#### 5. 训练方案二 Calibrator

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.fit_calibrator `
  --preds runs\rawalk_ego_scheme2_train_preds.csv `
  --out runs\rawalk_ego_scheme2_calibrator.joblib `
  --metrics-out runs\rawalk_ego_scheme2_calibrator.metrics.json `
  --model best `
  --group-column image_id
```

当前训练出的 calibrator 已复制到：

```text
models/rawalk_ego_scheme2_calibrator.joblib
```

#### 6. 方案二单图推理

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image path\to\image.jpg `
  --detector models\rawalk_yolo11s_960_e20_best.pt `
  --calibrator models\rawalk_ego_scheme2_calibrator.joblib `
  --out runs\scheme2_moge_calibrated.json `
  --vis runs\scheme2_moge_calibrated.png `
  --imgsz 960 `
  --geom-size 640 `
  --device cuda:0
```

方案二输出里会同时保留 MoGe raw 距离和 calibration 后的距离。最终看 `distance_calibrated_m`。

## 原理说明

### GT 是怎么计算的

Rawalk ego 距离 GT 来自数据集自带的 3D 信息：

```text
processed_data/fit_poses3d/*.npy
ego/ariaXX/calib/*.txt
ego/ariaXX/images/rgb/*.jpg
colmap/workplace/colmap_from_aria_transforms.pkl
```

计算流程：

1. 读取每一帧每个人的 3D 人体关键点。
2. 读取 ego Aria 相机的内参、外参和 Aria 到统一世界坐标的变换。
3. 把人体 3D 点从世界坐标系变换到当前相机坐标系。
4. 把相机坐标系下的 3D 点投影到图片上，得到 2D bbox。
5. 用相机坐标系下的点计算：

```text
depth_m = Z
distance_m = sqrt(X^2 + Y^2 + Z^2)
```

`depth_m` 是相机正前方方向的深度，`distance_m` 是相机中心到人的真实直线距离。

### MoGe 是做什么的

MoGe 是单目 3D 几何估计模型。输入一张 RGB 图片，输出整图 depth map 和相机 FOV/内参估计。

当前使用：

```text
Ruicheng/moge-2-vits-normal
```

在方案二里：

```text
YOLO 找人框
-> MoGe 给整图 depth map
-> 从人框/bbox 下方 60% 取 depth 中位数
-> 用相机内参反投影成 3D 点
-> distance_m = sqrt(X^2 + Y^2 + Z^2)
```

MoGe 不负责识别人，只负责估计图像几何。Rawalk 上 raw MoGe 有明显系统偏差，所以需要 calibration。

### Calibration 是怎么做的

Calibration 不是重新训练 YOLO，也不是重新训练 MoGe。它是一个小的后处理回归器，学习：

```text
MoGe raw distance/depth + 检测框/深度统计特征 -> Rawalk GT distance/depth
```

训练数据：

```text
runs/rawalk_ego_scheme2_train_preds.csv
```

输入特征：

| 输入特征 | 含义 |
|---|---|
| `z_depth_m` | MoGe 原始前向深度 |
| `distance_m` | MoGe 深度 + 内参算出来的原始直线距离 |
| `bbox_width_norm` / `bbox_height_norm` / `bbox_area_norm` | 人框尺寸 |
| `mask_area_norm` | mask 或 bbox 区域面积 |
| `center_x_norm` / `center_y_norm` | 人框中心位置 |
| `score` | YOLO 检测置信度 |
| `bearing_yaw_deg` / `elevation_pitch_deg` | 人相对相机的角度 |
| `fov_deg` | 相机水平视场角 |
| `depth_p10/p25/p50/p75/p90` | 人框/mask 区域内 MoGe depth 分位数 |
| `lower_depth_p10/p25/p50/p75/p90` | 人框/mask 下方 60% 区域内 MoGe depth 分位数 |

监督目标：

| 目标 | 含义 |
|---|---|
| `z_gt` | Rawalk 3D GT 算出来的真实前向深度 |
| `distance_gt` | Rawalk 3D GT 算出来的真实直线距离 |

当前保存的 calibrator 有两个回归器：

| 回归器 | 输出 |
|---|---|
| `z_model` | `z_depth_calibrated_m` |
| `distance_model` | `distance_calibrated_m` |

`fit_calibrator.py` 会尝试 `scale_bias`、`linear`、`ridge`、`gradient boosting`、`MLP`，在 train 内部按 `image_id` 做 held-out 验证，选择 distance MAE 最低的模型。当前 Rawalk scheme-2 选择的是 `MLP`。

## 代码入口

```text
human_detect.infer                     方案二单图推理：YOLO + MoGe + calibration
human_detect.infer_distance_head       方案一单图推理：YOLO + distance head
human_detect.prepare_rawalk_yolo       准备 Rawalk YOLO bbox 数据
human_detect.train_yolo                训练 YOLO person detector
human_detect.rawalk_ego_depth          从 Rawalk 3D pose + calib 生成 ego 距离 GT
human_detect.split_rawalk_ego_depth    固定切分 train/eval，并过滤异常距离
human_detect.train_yolo_distance_head  训练方案一 distance head
human_detect.eval_rawalk_ego_depth     跑方案二检测+深度并输出匹配预测
human_detect.fit_calibrator            训练 calibration regressor
```
