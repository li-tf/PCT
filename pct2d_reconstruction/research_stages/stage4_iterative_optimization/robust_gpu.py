"""Stage-local quadratic/Huber list-mode OS-SART projector."""

from __future__ import annotations

import numpy as np

from weighted_gpu import WeightedGpuMlpProjector


class RobustGpuMlpProjector(WeightedGpuMlpProjector):
    """Apply a frozen IRLS Huber factor to one OS-SART batch.

    The path construction and forward model are inherited unchanged.  For
    quadratic loss the factor is exactly one, so the update reproduces the
    retained equal-weight Stage-3 operator.
    """

    @staticmethod
    def huber_factors(residual, delta_mm: float | None, cp):
        if delta_mm is None:
            return cp.ones_like(residual, dtype=cp.float32)
        if not np.isfinite(delta_mm) or delta_mm <= 0.0:
            raise ValueError("Huber delta must be finite and positive")
        absolute = cp.abs(residual)
        return cp.minimum(
            cp.float32(1.0),
            cp.float32(delta_mm) / cp.maximum(absolute, cp.float32(1.0e-12)),
        )

    def accumulate_loss(
        self,
        image,
        batch: dict[str, np.ndarray],
        angle_deg: float,
        numerator,
        denominator,
        huber_delta_mm: float | None,
    ) -> dict[str, float | int]:
        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        if n == 0:
            return {
                "squared": 0.0,
                "absolute": 0.0,
                "signed": 0.0,
                "huber_objective": 0.0,
                "valid": 0,
            }
        (
            blocks,
            threads,
            pixels,
            path_weights,
            row_sum,
            normalized,
            residual_squared,
            valid,
        ) = self._paths_and_forward(image, batch, angle_deg)
        residual = normalized * row_sum
        factors = self.huber_factors(residual, huber_delta_mm, cp)
        factors *= valid
        self.weighted_back_kernel(
            blocks,
            (threads,),
            (
                pixels,
                path_weights,
                normalized,
                factors,
                valid,
                np.int32(n),
                np.int32(self.samples),
                numerator,
                denominator,
            ),
        )
        absolute = cp.abs(residual)
        if huber_delta_mm is None:
            objective = cp.float32(0.5) * residual_squared
        else:
            delta = cp.float32(huber_delta_mm)
            objective = cp.where(
                absolute <= delta,
                cp.float32(0.5) * residual_squared,
                delta * (absolute - cp.float32(0.5) * delta),
            )
        return {
            "squared": float(
                cp.sum(residual_squared * valid, dtype=cp.float64).get()
            ),
            "absolute": float(
                cp.sum(absolute * valid, dtype=cp.float64).get()
            ),
            "signed": float(
                cp.sum(residual * valid, dtype=cp.float64).get()
            ),
            "huber_objective": float(
                cp.sum(objective * valid, dtype=cp.float64).get()
            ),
            "valid": int(cp.sum(valid, dtype=cp.int64).get()),
        }
