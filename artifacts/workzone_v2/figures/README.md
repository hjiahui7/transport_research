# Workzone v2 论文图

本目录中的 PNG 为 300 DPI（定性案例图为 220 DPI），PDF 为矢量或混合矢量版本，适合论文排版。

`individual/` 保存全部单图版本。本目录不保留组合 PNG/PDF；主 README 通过表格并排引用这些单图，视觉布局保持不变。

## 单图文件

| 内容 | PNG | PDF |
|---|---|---|
| Depth pipeline 混淆矩阵 | [`fig1a`](individual/fig1a_depth_zone_confusion.png) | [`fig1a`](individual/fig1a_depth_zone_confusion.pdf) |
| VLM 直接估距混淆矩阵 | [`fig1b`](individual/fig1b_vlm_direct_zone_confusion.png) | [`fig1b`](individual/fig1b_vlm_direct_zone_confusion.pdf) |
| Raw MoGe 散点图 | [`fig2a`](individual/fig2a_raw_moge_scatter.png) | [`fig2a`](individual/fig2a_raw_moge_scatter.pdf) |
| Calibration 散点图 | [`fig2b`](individual/fig2b_calibrated_scatter.png) | [`fig2b`](individual/fig2b_calibrated_scatter.pdf) |
| 绝对误差 CDF | [`fig2c`](individual/fig2c_absolute_error_cdf.png) | [`fig2c`](individual/fig2c_absolute_error_cdf.pdf) |
| 近距离案例 | [`fig3a`](individual/fig3a_near_field_workers.png) | [`fig3a`](individual/fig3a_near_field_workers.pdf) |
| 局部遮挡案例 | [`fig3b`](individual/fig3b_partial_occlusion.png) | [`fig3b`](individual/fig3b_partial_occlusion.pdf) |
| Calibration 修正案例 | [`fig3c`](individual/fig3c_calibration_correction.png) | [`fig3c`](individual/fig3c_calibration_correction.pdf) |
| 三人多区域案例 | [`fig3d`](individual/fig3d_multiple_workers_and_zones.png) | [`fig3d`](individual/fig3d_multiple_workers_and_zones.pdf) |
| VLM 属性准确率 | [`fig4`](individual/fig4_vlm_attribute_accuracy.png) | [`fig4`](individual/fig4_vlm_attribute_accuracy.pdf) |

## 图文件

- `fig1a` 和 `fig1b`：MoGe + calibration 与 VLM 直接估距的距离区域混淆矩阵。两图严格使用相同的 332 个 worker；红框表示 GT 为 Danger、但被预测成更安全区域的关键漏判。
- `fig2a`、`fig2b` 和 `fig2c`：前两张散点图使用全部 332 个距离匹配；CDF 使用四种方案共同具备结果的 330 个 worker。
- `fig3a` 至 `fig3d`：近距离、局部遮挡、calibration 修正和多人多区域四类案例。其中遮挡案例为 `Pave1_001686`，多人多区域案例为 `Garage4_001530`。框颜色为红色 Danger、黄色 Caution、绿色 Lower-risk。
- `fig4`：四个 VLM 结果在共同匹配的 405 个 worker 上比较；每个属性计算时排除该属性 GT 为 `uncertain` 的 worker。
- `figure_metrics.json`：所有图中的矩阵、样本数和精确指标，便于核对论文正文。

## 关键数字

- 距离区域准确率：MoGe + calibration `92.5%`，VLM 直接估距 `56.6%`。
- Danger recall：MoGe + calibration `94.8%`，VLM 直接估距 `28.6%`。
- Raw MoGe：MAE `1.856m`，bias `+1.720m`。
- MoGe + MLP calibration：MAE `0.334m`，bias `-0.197m`。

## 重新生成

在项目根目录执行：

```powershell
D:\coding\anaconda\envs\qwen\python.exe scripts\plot_workzone_paper_figures.py
```

数值图只依赖 `artifacts/workzone_v2/results/`。定性案例图还需要原始数据集图片位于 `work-zone-safety-rgbd-dataset/images/`。
