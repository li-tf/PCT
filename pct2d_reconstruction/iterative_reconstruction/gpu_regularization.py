"""GPU proximal TV and Huber-TV regularization for iterative pCT."""

from __future__ import annotations

from collections.abc import Callable
import time


ProgressCallback = Callable[[int, int, float], None]


def _forward_gradient(cp, image, edge_x, edge_z):
    gx = cp.zeros_like(image)
    gz = cp.zeros_like(image)
    gx[:, :-1] = (image[:, 1:] - image[:, :-1]) * edge_x
    gz[:-1, :] = (image[1:, :] - image[:-1, :]) * edge_z
    return gx, gz


def _divergence(cp, px, pz):
    """Return ``-K.T p`` for the forward-difference gradient K."""

    result = cp.zeros_like(px)
    result[:, :-1] += px[:, :-1]
    result[:, 1:] -= px[:, :-1]
    result[:-1, :] += pz[:-1, :]
    result[1:, :] -= pz[:-1, :]
    return result


def regularization_value(image, support, kind: str, weight: float, huber_delta: float) -> float:
    """Evaluate weighted isotropic TV/Huber-TV inside the support."""

    cp = __import__("cupy")
    edge_x = support[:, :-1] & support[:, 1:]
    edge_z = support[:-1, :] & support[1:, :]
    gx, gz = _forward_gradient(cp, image, edge_x, edge_z)
    magnitude = cp.sqrt(gx * gx + gz * gz)
    if kind == "huber_tv":
        density = cp.where(
            magnitude <= huber_delta,
            0.5 * magnitude * magnitude / huber_delta,
            magnitude - 0.5 * huber_delta,
        )
    else:
        density = magnitude
    return float((weight * cp.sum(density, dtype=cp.float64)).get())


def proximal_regularize(
    image,
    support,
    *,
    kind: str,
    weight: float,
    iterations: int,
    huber_delta: float = 0.01,
    primal_step: float = 0.25,
    dual_step: float = 0.25,
    extrapolation: float = 1.0,
    progress_callback: ProgressCallback | None = None,
):
    """Apply the ROF proximal operator using Chambolle--Pock iterations.

    Solves ``0.5*||u-f||^2 + weight*R(u)`` subject to non-negativity and the
    supplied support. ``R`` is isotropic TV or its Huber-smoothed form. Image
    differences crossing the known circular support boundary are excluded so
    regularization does not shrink the water-cylinder edge toward zero.
    """

    if kind not in {"tv", "huber_tv"}:
        raise ValueError("regularizer must be 'tv' or 'huber_tv'")
    if weight <= 0.0 or iterations < 1 or huber_delta <= 0.0:
        raise ValueError("regularization weight, iterations, and Huber delta must be positive")
    if primal_step <= 0.0 or dual_step <= 0.0:
        raise ValueError("primal and dual steps must be positive")
    # ||K||^2 <= 8 for a 2-D forward-difference gradient.
    if 8.0 * primal_step * dual_step >= 1.0:
        raise ValueError("require 8 * primal_step * dual_step < 1 for convergence")

    cp = __import__("cupy")
    started = time.perf_counter()
    reference = image.copy()
    u = reference.copy()
    u_bar = u.copy()
    px = cp.zeros_like(u)
    pz = cp.zeros_like(u)
    edge_x = support[:, :-1] & support[:, 1:]
    edge_z = support[:-1, :] & support[1:, :]
    report_every = max(1, iterations // 5)

    for iteration in range(1, iterations + 1):
        gx, gz = _forward_gradient(cp, u_bar, edge_x, edge_z)
        px += dual_step * gx
        pz += dual_step * gz
        if kind == "huber_tv":
            # prox of the quadratic part of the Huber-TV conjugate.
            dual_scale = 1.0 + dual_step * huber_delta / weight
            px /= dual_scale
            pz /= dual_scale
        dual_norm = cp.sqrt(px * px + pz * pz)
        projection = cp.maximum(1.0, dual_norm / weight)
        px /= projection
        pz /= projection

        previous = u
        # _divergence is -K.T, hence the plus sign below.
        candidate = u + primal_step * _divergence(cp, px, pz)
        u = (candidate + primal_step * reference) / (1.0 + primal_step)
        cp.maximum(u, 0.0, out=u)
        u *= support
        u_bar = u + extrapolation * (u - previous)

        if progress_callback is not None and (
            iteration == 1 or iteration % report_every == 0 or iteration == iterations
        ):
            relative_change = float(
                (cp.linalg.norm(u - previous) / cp.maximum(cp.linalg.norm(previous), 1.0e-20)).get()
            )
            progress_callback(iteration, iterations, relative_change)

    cp.cuda.Stream.null.synchronize()
    difference = u - reference
    metrics = {
        "kind": kind,
        "weight": float(weight),
        "iterations": int(iterations),
        "huber_delta": float(huber_delta),
        "primal_step": float(primal_step),
        "dual_step": float(dual_step),
        "l2_change": float(cp.linalg.norm(difference).get()),
        "max_abs_change": float(cp.max(cp.abs(difference)).get()),
        "regularization_value_before": regularization_value(
            reference, support, kind, weight, huber_delta
        ),
        "regularization_value_after": regularization_value(
            u, support, kind, weight, huber_delta
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return u, metrics
