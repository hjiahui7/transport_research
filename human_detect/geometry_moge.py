from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import CameraIntrinsics, fov_x_from_intrinsics, intrinsics_from_normalized_matrix


@dataclass(frozen=True)
class GeometryResult:
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    fov_deg: float | None
    valid_mask: np.ndarray | None = None


class MogeGeometry:
    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vits-normal",
        *,
        device: str = "cuda:0",
        half: bool = True,
        geom_size: int = 768,
        num_tokens: int | None = 1200,
    ) -> None:
        try:
            import torch
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise RuntimeError("MoGe is not installed. Install git+https://github.com/microsoft/MoGe.git in the qwen conda environment.") from exc

        self.torch = torch
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self.geom_size = geom_size
        self.num_tokens = num_tokens
        self.model = MoGeModel.from_pretrained(model_name).to(self.device).eval()

    def infer(self, image_path: str | Path) -> GeometryResult:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        resized_rgb, scale = _resize_long_edge(image_rgb, self.geom_size)
        resized_h, resized_w = resized_rgb.shape[:2]

        image_tensor = self.torch.tensor(resized_rgb / 255.0, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        with self.torch.inference_mode():
            with self.torch.autocast(device_type=self.device.type, dtype=self.torch.float16, enabled=self.half):
                output = self._infer_model(image_tensor)

        depth = _to_numpy(output["depth"]).astype(np.float32)
        if depth.shape != (resized_h, resized_w):
            depth = cv2.resize(depth, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        if (resized_h, resized_w) != (original_h, original_w):
            depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        intrinsics_value = output.get("intrinsics")
        if intrinsics_value is None:
            raise RuntimeError("MoGe output did not include intrinsics")
        intrinsics_resized = intrinsics_from_normalized_matrix(_to_numpy(intrinsics_value), resized_w, resized_h)
        intrinsics = intrinsics_resized.scaled(
            sx=original_w / resized_w,
            sy=original_h / resized_h,
            width=original_w,
            height=original_h,
        )
        fov_deg = fov_x_from_intrinsics(intrinsics, original_w)

        valid_mask = output.get("mask")
        valid_np = None
        if valid_mask is not None:
            valid_np = _to_numpy(valid_mask).astype(bool)
            if valid_np.shape != (resized_h, resized_w):
                valid_np = cv2.resize(valid_np.astype(np.uint8), (resized_w, resized_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if valid_np.shape != (original_h, original_w):
                valid_np = cv2.resize(valid_np.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return GeometryResult(depth=depth, intrinsics=intrinsics, fov_deg=fov_deg, valid_mask=valid_np)

    def _infer_model(self, image_tensor):
        signature = inspect.signature(self.model.infer)
        kwargs = {}
        if self.num_tokens is not None and "num_tokens" in signature.parameters:
            kwargs["num_tokens"] = self.num_tokens
        elif "resolution_level" in signature.parameters:
            kwargs["resolution_level"] = 5
        try:
            return self.model.infer(image_tensor, **kwargs)
        except TypeError:
            return self.model.infer(image_tensor)


def _resize_long_edge(image: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    if long_edge <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    current_long_edge = max(height, width)
    if current_long_edge <= long_edge:
        return image, 1.0
    scale = long_edge / current_long_edge
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)

