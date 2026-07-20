# Qwen VLM Prompt

实际调用代码在：

```text
human_detect/workzone_report.py
```

- 单图 prompt：`call_qwen_visual_attributes`
- 批量 prompt：`_batch_prompt`
- system message：`Return strict JSON only. No markdown.`

## 单图 Prompt

```text
You are labeling construction worker visual attributes from one annotated image.
The image has red boxes with labels like W1, W2. Use those labels as worker_index.

Workers:
{worker_lines}

Return JSON only with this schema:
{
  "workers": [
    {
      "worker_index": 1,
      "high_visibility_vest": true | false | "uncertain",
      "helmet_status": "worn" | "absent" | "uncertain",
      "orientation": "Facing" | "Side" | "Back" | "uncertain",
      "occlusion_level": "none" | "partial" | "heavy" | "uncertain"
    }
  ]
}

Rules:
- high_visibility_vest=true only when a high-visibility vest or jacket is clearly visible.
- helmet_status=worn only when a helmet is on the worker's head. Helmet in hand means absent.
- orientation is relative to camera view.
- Do not estimate distance.
```

`{worker_lines}` 由代码自动填，例如：

```text
- worker_index=1, bbox_xyxy=[100.2, 40.5, 220.1, 300.0]
- worker_index=2, bbox_xyxy=[260.4, 70.0, 390.2, 310.8]
```

## Batch Prompt

```text
You are labeling construction worker visual attributes from multiple annotated images.
Each image is preceded by a text marker IMAGE_ID. The image itself has red boxes with labels like W1, W2.
Use the red W number as worker_index for that image.

Images and workers:
{worker_block}

Return JSON only with this schema:
{
  "images": [
    {
      "image_id": "image file name from IMAGE_ID",
      "workers": [
        {
          "worker_index": 1,
          "high_visibility_vest": true | false | "uncertain",
          "helmet_status": "worn" | "absent" | "uncertain",
          "orientation": "Facing" | "Side" | "Back" | "uncertain",
          "occlusion_level": "none" | "partial" | "heavy" | "uncertain"
        }
      ]
    }
  ]
}

Rules:
- high_visibility_vest=true only when a high-visibility vest or jacket is clearly visible.
- helmet_status=worn only when a helmet is on the worker's head. Helmet in hand means absent.
- orientation is relative to camera view.
- Do not estimate distance.
- Include every image_id and every worker_index listed above.
```

`{worker_block}` 由代码自动填，例如：

```text
image_id=Garage1_000840.png
- worker_index=1, bbox_xyxy=[100.2, 40.5, 220.1, 300.0]
image_id=Garage1_000888.png
- worker_index=1, bbox_xyxy=[260.4, 70.0, 390.2, 310.8]
```

## 重要约束

距离不交给 Qwen 猜。Qwen 只判断下面四个视觉属性：

```text
high_visibility_vest
helmet_status
orientation
occlusion_level
```

最终 JSON 里的 `distance_to_equipment_m` 来自本地 distance head 或 MoGe/calibration，`distance_band` 由代码按米数自动生成。
