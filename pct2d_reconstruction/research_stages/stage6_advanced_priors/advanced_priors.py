"""GPU proximal operators used only by Stage 6.

The operators solve an image-domain proximal problem after each fixed-MLP
OS-SART epoch.  They deliberately do not change the pCT path or data model.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any


ProgressCallback = Callable[[int, int, float], None]


def forward_gradient(xp, image, edge_x, edge_z):
    gx = xp.zeros_like(image)
    gz = xp.zeros_like(image)
    gx[:, :-1] = (image[:, 1:] - image[:, :-1]) * edge_x
    gz[:-1, :] = (image[1:, :] - image[:-1, :]) * edge_z
    return gx, gz


def negative_gradient_adjoint(xp, px, pz, edge_x=None, edge_z=None):
    """Return ``-G.T @ (px, pz)`` for the masked forward gradient."""

    if edge_x is not None:
        px = px.copy()
        px[:, :-1] *= edge_x
    if edge_z is not None:
        pz = pz.copy()
        pz[:-1, :] *= edge_z
    result = xp.zeros_like(px)
    result[:, :-1] += px[:, :-1]
    result[:, 1:] -= px[:, :-1]
    result[:-1, :] += pz[:-1, :]
    result[1:, :] -= pz[:-1, :]
    return result


def symmetric_gradient(xp, vx, vz, edge_x, edge_z):
    """Symmetric gradient with Frobenius convention for the cross term."""

    dvx_dx, dvx_dz = forward_gradient(xp, vx, edge_x, edge_z)
    dvz_dx, dvz_dz = forward_gradient(xp, vz, edge_x, edge_z)
    return dvx_dx, dvz_dz, 0.5 * (dvx_dz + dvz_dx)


def negative_symmetric_gradient_adjoint(
    xp, qxx, qzz, qxz, edge_x, edge_z
):
    """Return ``-E.T @ q`` under ``<Ev,q>=xx+zz+2*xz``."""

    out_x = negative_gradient_adjoint(
        xp, qxx, qxz, edge_x=edge_x, edge_z=edge_z
    )
    out_z = negative_gradient_adjoint(
        xp, qxz, qzz, edge_x=edge_x, edge_z=edge_z
    )
    return out_x, out_z


def _smooth_five_point(xp, image, support, iterations: int):
    value = image.copy()
    support_f = support.astype(image.dtype)
    for _ in range(max(0, int(iterations))):
        total = 4.0 * value
        count = 4.0 * support_f
        total[:, 1:] += value[:, :-1] * support_f[:, :-1]
        count[:, 1:] += support_f[:, :-1]
        total[:, :-1] += value[:, 1:] * support_f[:, 1:]
        count[:, :-1] += support_f[:, 1:]
        total[1:, :] += value[:-1, :] * support_f[:-1, :]
        count[1:, :] += support_f[:-1, :]
        total[:-1, :] += value[1:, :] * support_f[1:, :]
        count[:-1, :] += support_f[1:, :]
        value = xp.where(support, total / xp.maximum(count, 1.0), 0.0)
    return value


def guidance_weights(
    xp,
    reference,
    support,
    *,
    mode: str,
    minimum_weight: float,
    smoothing_iterations: int,
    kappa_quantile: float,
):
    """Create fixed scalar or direction-dependent edge weights."""

    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in (0, 1]")
    if not 0.0 < kappa_quantile < 1.0:
        raise ValueError("kappa_quantile must be in (0, 1)")
    edge_x = support[:, :-1] & support[:, 1:]
    edge_z = support[:-1, :] & support[1:, :]
    smooth = _smooth_five_point(
        xp, reference, support, smoothing_iterations
    )
    gx, gz = forward_gradient(xp, smooth, edge_x, edge_z)
    magnitude = xp.sqrt(gx * gx + gz * gz)
    samples = magnitude[support]
    positive = samples[samples > 0]
    kappa = (
        xp.quantile(positive, kappa_quantile)
        if int(positive.size)
        else xp.asarray(1.0e-3, dtype=reference.dtype)
    )
    kappa = xp.maximum(kappa, xp.asarray(1.0e-6, dtype=reference.dtype))
    if mode == "adaptive":
        scalar = minimum_weight + (1.0 - minimum_weight) / (
            1.0 + (magnitude / kappa) ** 2
        )
        wx = scalar
        wz = scalar
    elif mode == "directional":
        wx = minimum_weight + (1.0 - minimum_weight) / (
            1.0 + (xp.abs(gx) / kappa) ** 2
        )
        wz = minimum_weight + (1.0 - minimum_weight) / (
            1.0 + (xp.abs(gz) / kappa) ** 2
        )
    else:
        raise ValueError(f"unknown guidance mode: {mode}")
    return wx, wz, float(kappa.item())


def _weighted_huber_tv(
    image,
    support,
    *,
    weight: float,
    huber_delta: float,
    iterations: int,
    primal_step: float,
    dual_step: float,
    reference_guidance,
    guidance_mode: str,
    minimum_weight: float,
    smoothing_iterations: int,
    kappa_quantile: float,
    progress_callback: ProgressCallback | None,
):
    cp = __import__("cupy")
    reference = image.copy()
    u = reference.copy()
    u_bar = u.copy()
    px = cp.zeros_like(u)
    pz = cp.zeros_like(u)
    edge_x = support[:, :-1] & support[:, 1:]
    edge_z = support[:-1, :] & support[1:, :]
    wx, wz, kappa = guidance_weights(
        cp,
        reference_guidance,
        support,
        mode=guidance_mode,
        minimum_weight=minimum_weight,
        smoothing_iterations=smoothing_iterations,
        kappa_quantile=kappa_quantile,
    )
    radius_x = cp.maximum(weight * wx, 1.0e-8)
    radius_z = cp.maximum(weight * wz, 1.0e-8)
    report_every = max(1, iterations // 5)
    for iteration in range(1, iterations + 1):
        gx, gz = forward_gradient(cp, u_bar, edge_x, edge_z)
        px += dual_step * gx
        pz += dual_step * gz
        px /= 1.0 + dual_step * huber_delta / radius_x
        pz /= 1.0 + dual_step * huber_delta / radius_z
        normalized = cp.sqrt(
            (px / radius_x) ** 2 + (pz / radius_z) ** 2
        )
        projection = cp.maximum(1.0, normalized)
        px /= projection
        pz /= projection
        previous = u
        candidate = u + primal_step * negative_gradient_adjoint(
            cp, px, pz, edge_x=edge_x, edge_z=edge_z
        )
        u = (candidate + primal_step * reference) / (1.0 + primal_step)
        cp.maximum(u, 0.0, out=u)
        u *= support
        u_bar = u + (u - previous)
        if progress_callback and (
            iteration == 1
            or iteration % report_every == 0
            or iteration == iterations
        ):
            change = cp.linalg.norm(u - previous) / cp.maximum(
                cp.linalg.norm(previous), 1.0e-20
            )
            progress_callback(iteration, iterations, float(change.get()))
    return u, {"guidance_kappa": kappa}


def _huber_tgv(
    image,
    support,
    *,
    weight: float,
    second_order_ratio: float,
    huber_delta: float,
    iterations: int,
    primal_step: float,
    dual_step: float,
    progress_callback: ProgressCallback | None,
):
    cp = __import__("cupy")
    reference = image.copy()
    u = reference.copy()
    u_bar = u.copy()
    vx = cp.zeros_like(u)
    vz = cp.zeros_like(u)
    vx_bar = vx.copy()
    vz_bar = vz.copy()
    px = cp.zeros_like(u)
    pz = cp.zeros_like(u)
    qxx = cp.zeros_like(u)
    qzz = cp.zeros_like(u)
    qxz = cp.zeros_like(u)
    edge_x = support[:, :-1] & support[:, 1:]
    edge_z = support[:-1, :] & support[1:, :]
    alpha1 = float(weight)
    alpha0 = float(weight * second_order_ratio)
    report_every = max(1, iterations // 5)
    for iteration in range(1, iterations + 1):
        gx, gz = forward_gradient(cp, u_bar, edge_x, edge_z)
        px += dual_step * (gx - vx_bar)
        pz += dual_step * (gz - vz_bar)
        px /= 1.0 + dual_step * huber_delta / alpha1
        pz /= 1.0 + dual_step * huber_delta / alpha1
        pnorm = cp.sqrt(px * px + pz * pz)
        pscale = cp.maximum(1.0, pnorm / alpha1)
        px /= pscale
        pz /= pscale

        exx, ezz, exz = symmetric_gradient(
            cp, vx_bar, vz_bar, edge_x, edge_z
        )
        qxx += dual_step * exx
        qzz += dual_step * ezz
        qxz += dual_step * exz
        qxx /= 1.0 + dual_step * huber_delta / alpha0
        qzz /= 1.0 + dual_step * huber_delta / alpha0
        qxz /= 1.0 + dual_step * huber_delta / alpha0
        qnorm = cp.sqrt(qxx * qxx + qzz * qzz + 2.0 * qxz * qxz)
        qscale = cp.maximum(1.0, qnorm / alpha0)
        qxx /= qscale
        qzz /= qscale
        qxz /= qscale

        previous_u = u
        previous_vx = vx
        previous_vz = vz
        u_candidate = u + primal_step * negative_gradient_adjoint(
            cp, px, pz, edge_x=edge_x, edge_z=edge_z
        )
        u = (u_candidate + primal_step * reference) / (
            1.0 + primal_step
        )
        cp.maximum(u, 0.0, out=u)
        u *= support
        neg_ex, neg_ez = negative_symmetric_gradient_adjoint(
            cp, qxx, qzz, qxz, edge_x, edge_z
        )
        vx += primal_step * (px + neg_ex)
        vz += primal_step * (pz + neg_ez)
        vx *= support
        vz *= support
        u_bar = u + (u - previous_u)
        vx_bar = vx + (vx - previous_vx)
        vz_bar = vz + (vz - previous_vz)
        if progress_callback and (
            iteration == 1
            or iteration % report_every == 0
            or iteration == iterations
        ):
            change = cp.linalg.norm(u - previous_u) / cp.maximum(
                cp.linalg.norm(previous_u), 1.0e-20
            )
            progress_callback(iteration, iterations, float(change.get()))
    return u, {"second_order_weight": alpha0}


def proximal_advanced(
    image,
    support,
    *,
    method: str,
    weight: float,
    iterations: int,
    huber_delta: float,
    primal_step: float,
    dual_step: float,
    reference_guidance=None,
    second_order_ratio: float = 1.0,
    minimum_weight: float = 0.3,
    smoothing_iterations: int = 4,
    kappa_quantile: float = 0.90,
    progress_callback: ProgressCallback | None = None,
):
    """Apply one of the Stage-6 proximal image priors."""

    if method not in {"tgv", "adaptive_tv", "directional_tv"}:
        raise ValueError(f"unknown prior: {method}")
    if weight <= 0 or iterations < 1 or huber_delta <= 0:
        raise ValueError("weight, iterations and Huber delta must be positive")
    if primal_step <= 0 or dual_step <= 0:
        raise ValueError("primal and dual steps must be positive")
    if method == "tgv" and second_order_ratio <= 0:
        raise ValueError("second_order_ratio must be positive")
    cp = __import__("cupy")
    started = time.perf_counter()
    reference = image.copy()
    if method == "tgv":
        result, extra = _huber_tgv(
            image,
            support,
            weight=weight,
            second_order_ratio=second_order_ratio,
            huber_delta=huber_delta,
            iterations=iterations,
            primal_step=primal_step,
            dual_step=dual_step,
            progress_callback=progress_callback,
        )
    else:
        result, extra = _weighted_huber_tv(
            image,
            support,
            weight=weight,
            huber_delta=huber_delta,
            iterations=iterations,
            primal_step=primal_step,
            dual_step=dual_step,
            reference_guidance=(
                reference if reference_guidance is None else reference_guidance
            ),
            guidance_mode=(
                "adaptive" if method == "adaptive_tv" else "directional"
            ),
            minimum_weight=minimum_weight,
            smoothing_iterations=smoothing_iterations,
            kappa_quantile=kappa_quantile,
            progress_callback=progress_callback,
        )
    cp.cuda.Stream.null.synchronize()
    difference = result - reference
    metrics: dict[str, Any] = {
        "method": method,
        "weight": float(weight),
        "iterations": int(iterations),
        "huber_delta": float(huber_delta),
        "l2_change": float(cp.linalg.norm(difference).get()),
        "max_abs_change": float(cp.max(cp.abs(difference)).get()),
        "elapsed_seconds": time.perf_counter() - started,
        **extra,
    }
    return result, metrics
