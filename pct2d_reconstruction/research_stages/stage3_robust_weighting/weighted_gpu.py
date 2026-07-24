"""Stage-local weighted extension of the retained Schulte-MLP CUDA operator."""

from __future__ import annotations

import numpy as np

from gpu_mlp_operator import GpuMlpProjector


WEIGHTED_BACKPROJECT = r"""
extern "C" __global__ void backproject_rows_weighted(
    const int *pixels, const float *path_weights, const float *normalized,
    const float *data_weights, const unsigned char *valid, int rays, int samples,
    float *numerator, float *denominator) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;
  if(ray>=rays || !valid[ray])return;
  float data_weight=data_weights[ray];
  float value=normalized[ray]*data_weight;
  int begin=ray*samples*4,end=begin+samples*4;
  for(int k=begin;k<end;++k){
    int p=pixels[k];
    if(p>=0){
      float spatial=path_weights[k];
      atomicAdd(numerator+p,spatial*value);
      atomicAdd(denominator+p,spatial*data_weight);
    }
  }
}
"""


class WeightedGpuMlpProjector(GpuMlpProjector):
    """Use identical MLP/path weights but multiply both normal equations by W."""

    def __init__(
        self, size: int, spacing_mm: float, step_mm: float, radius_mm: float
    ):
        super().__init__(size, spacing_mm, step_mm, radius_mm)
        module = self.cp.RawModule(
            code=WEIGHTED_BACKPROJECT, options=("--std=c++11",)
        )
        self.weighted_back_kernel = module.get_function(
            "backproject_rows_weighted"
        )

    def _paths_and_forward(self, image, batch, angle_deg):
        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[name], dtype=np.float32))
            for name in (
                "position_in",
                "position_out",
                "direction_in",
                "direction_out",
            )
        ]
        wepl = cp.asarray(
            np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32)
        )
        entries = n * self.samples * 4
        pixels = cp.empty(entries, dtype=cp.int32)
        path_weights = cp.empty(entries, dtype=cp.float32)
        row_sum = cp.empty(n, dtype=cp.float32)
        normalized = cp.empty(n, dtype=cp.float32)
        residual_squared = cp.empty(n, dtype=cp.float32)
        valid = cp.empty(n, dtype=cp.uint8)
        threads = 128
        blocks = ((n + threads - 1) // threads,)
        angle = np.deg2rad(angle_deg)
        self.build_kernel(
            blocks,
            (threads,),
            (
                *inputs,
                np.int32(n),
                np.int32(self.samples),
                np.float64(-self.radius + 0.5 * self.step),
                np.float64(self.step),
                np.float64(self.radius),
                np.int32(self.size),
                np.float64(self.spacing),
                np.float64(self.origin),
                np.float64(np.cos(angle)),
                np.float64(np.sin(angle)),
                pixels,
                path_weights,
                row_sum,
            ),
        )
        self.forward_kernel(
            blocks,
            (threads,),
            (
                image,
                pixels,
                path_weights,
                row_sum,
                wepl,
                np.int32(n),
                np.int32(self.samples),
                normalized,
                residual_squared,
                valid,
            ),
        )
        return (
            blocks,
            threads,
            pixels,
            path_weights,
            row_sum,
            normalized,
            residual_squared,
            valid,
        )

    def accumulate_weighted(
        self,
        image,
        batch: dict[str, np.ndarray],
        data_weights: np.ndarray,
        angle_deg: float,
        numerator,
        denominator,
    ) -> dict[str, float | int]:
        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        if n == 0:
            return {
                "unweighted_squared": 0.0,
                "weighted_squared": 0.0,
                "weight_sum": 0.0,
                "valid": 0,
            }
        data_weights = np.ascontiguousarray(data_weights, dtype=np.float32)
        if len(data_weights) != n or not np.isfinite(data_weights).all():
            raise ValueError("data weights must be finite and match the batch")
        (
            blocks,
            threads,
            pixels,
            path_weights,
            _row_sum,
            normalized,
            residual_squared,
            valid,
        ) = self._paths_and_forward(image, batch, angle_deg)
        device_weights = cp.asarray(data_weights)
        self.weighted_back_kernel(
            blocks,
            (threads,),
            (
                pixels,
                path_weights,
                normalized,
                device_weights,
                valid,
                np.int32(n),
                np.int32(self.samples),
                numerator,
                denominator,
            ),
        )
        valid_weights = device_weights * valid
        return {
            "unweighted_squared": float(
                cp.sum(residual_squared, dtype=cp.float64).get()
            ),
            "weighted_squared": float(
                cp.sum(
                    residual_squared * valid_weights, dtype=cp.float64
                ).get()
            ),
            "weight_sum": float(cp.sum(valid_weights, dtype=cp.float64).get()),
            "valid": int(cp.sum(valid, dtype=cp.int64).get()),
        }

    def residuals(
        self, image, batch: dict[str, np.ndarray], angle_deg: float
    ) -> np.ndarray:
        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        if n == 0:
            return np.empty(0, dtype=np.float32)
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[name], dtype=np.float32))
            for name in (
                "position_in",
                "position_out",
                "direction_in",
                "direction_out",
            )
        ]
        wepl = cp.asarray(
            np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32)
        )
        entries = n * self.samples * 4
        pixels = cp.empty(entries, dtype=cp.int32)
        path_weights = cp.empty(entries, dtype=cp.float32)
        row_sum = cp.empty(n, dtype=cp.float32)
        residual = cp.empty(n, dtype=cp.float32)
        squared = cp.empty(n, dtype=cp.float32)
        absolute = cp.empty(n, dtype=cp.float32)
        valid = cp.empty(n, dtype=cp.uint8)
        threads = 128
        blocks = ((n + threads - 1) // threads,)
        angle = np.deg2rad(angle_deg)
        self.build_kernel(
            blocks,
            (threads,),
            (
                *inputs,
                np.int32(n),
                np.int32(self.samples),
                np.float64(-self.radius + 0.5 * self.step),
                np.float64(self.step),
                np.float64(self.radius),
                np.int32(self.size),
                np.float64(self.spacing),
                np.float64(self.origin),
                np.float64(np.cos(angle)),
                np.float64(np.sin(angle)),
                pixels,
                path_weights,
                row_sum,
            ),
        )
        self.evaluate_kernel(
            blocks,
            (threads,),
            (
                image,
                pixels,
                path_weights,
                row_sum,
                wepl,
                np.int32(n),
                np.int32(self.samples),
                residual,
                squared,
                absolute,
                valid,
            ),
        )
        return cp.asnumpy(residual[valid.astype(cp.bool_)])
