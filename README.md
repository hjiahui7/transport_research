# human_detect

Lightweight v0 pipeline for single-image multi-person camera-distance and viewing-angle inference.

## Environment

This workspace is set up to use the existing conda environment:

```powershell
D:\coding\anaconda\envs\qwen\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The current implementation uses:

- `YOLO11n-seg` for COCO `person` instance segmentation.
- `Ruicheng/moge-2-vits-normal` for metric depth and camera intrinsics/FOV.
- PM-HMCW KITTI-style `image_2`, `label_2`, and `calib` files for parser tests and optional intrinsics.

## Data Layout

The v0 PM-HMCW data is expected at:

```text
data/pm_hmcw/raw/real-world/test/{image_2,label_2,calib}
data/pm_hmcw/raw/virtual/test/{image_2,label_2,calib}
```

`data/` and `runs/` are ignored by git.

## Inference

With PM-HMCW calibration:

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

Without calibration, the CLI uses MoGe intrinsics:

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image data\pm_hmcw\raw\real-world\test\image_2\000248.png `
  --out runs\smoke_000248_no_calib.json `
  --vis runs\smoke_000248_no_calib.png
```

## Tests

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m pytest -q
```

## Calibration

Generate matched PM-HMCW prediction rows:

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.eval_pm_hmcw `
  --data data\pm_hmcw\raw `
  --split-contains real-world\test `
  --out runs\pm_hmcw_real100_preds.csv `
  --imgsz 640 `
  --geom-size 640 `
  --device cuda:0 `
  --half
```

Fit and save a lightweight calibrator:

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.fit_calibrator `
  --preds runs\pm_hmcw_real100_preds.csv `
  --out runs\calibrator_real100_group.joblib `
  --metrics-out runs\calibrator_real100_group.metrics.json `
  --model best `
  --group-column image_id
```

Use it during inference:

```powershell
D:\coding\anaconda\envs\qwen\python.exe -m human_detect.infer `
  --image data\pm_hmcw\raw\real-world\test\image_2\000248.png `
  --calib data\pm_hmcw\raw\real-world\test\calib\000248.txt `
  --calibrator runs\calibrator_real100_group.joblib `
  --out runs\smoke_000248_calibrated_group.json `
  --vis runs\smoke_000248_calibrated_group.png
```

The calibrator is a post-processing regressor. It does not train YOLO or MoGe.
On the current 100-image real-world split, grouped validation distance MAE changed from `0.926m` raw to `0.455m` with the selected Ridge correction head using bbox/mask/FOV/depth-quantile features. Individual images can still get worse, so keep both raw and calibrated fields in the JSON.
