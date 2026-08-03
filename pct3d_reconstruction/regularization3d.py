"""Three-dimensional support-aware Huber-TV proximal operator."""

from __future__ import annotations


def _gradient(cp, u, ex, ey, ez):
    gx, gy, gz = cp.zeros_like(u), cp.zeros_like(u), cp.zeros_like(u)
    gx[:, :, :-1] = (u[:, :, 1:] - u[:, :, :-1]) * ex
    gy[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) * ey
    gz[:-1, :, :] = (u[1:, :, :] - u[:-1, :, :]) * ez
    return gx, gy, gz


def _divergence(cp, px, py, pz):
    result = cp.zeros_like(px)
    result[:, :, :-1] += px[:, :, :-1]
    result[:, :, 1:] -= px[:, :, :-1]
    result[:, :-1, :] += py[:, :-1, :]
    result[:, 1:, :] -= py[:, :-1, :]
    result[:-1, :, :] += pz[:-1, :, :]
    result[1:, :, :] -= pz[:-1, :, :]
    return result


def proximal_huber_tv(image, support, weight: float, delta: float, iterations: int):
    if weight == 0:
        return image, {"weight": 0.0, "iterations": 0, "l2_change": 0.0}
    import cupy as cp

    primal_step = dual_step = 0.2
    reference = image.copy()
    u = reference.copy()
    ubar = u.copy()
    px, py, pz = cp.zeros_like(u), cp.zeros_like(u), cp.zeros_like(u)
    ex = support[:, :, :-1] & support[:, :, 1:]
    ey = support[:, :-1, :] & support[:, 1:, :]
    ez = support[:-1, :, :] & support[1:, :, :]
    for _ in range(iterations):
        gx, gy, gz = _gradient(cp, ubar, ex, ey, ez)
        px += dual_step * gx
        py += dual_step * gy
        pz += dual_step * gz
        scale = 1.0 + dual_step * delta / weight
        px /= scale
        py /= scale
        pz /= scale
        norm = cp.sqrt(px * px + py * py + pz * pz)
        projection = cp.maximum(1.0, norm / weight)
        px /= projection
        py /= projection
        pz /= projection
        previous = u
        candidate = u + primal_step * _divergence(cp, px, py, pz)
        u = (candidate + primal_step * reference) / (1.0 + primal_step)
        cp.maximum(u, 0.0, out=u)
        u *= support
        ubar = 2.0 * u - previous
    difference = u - reference
    return u, {
        "weight": float(weight),
        "iterations": int(iterations),
        "l2_change": float(cp.linalg.norm(difference).get()),
        "max_abs_change": float(cp.max(cp.abs(difference)).get()),
    }
