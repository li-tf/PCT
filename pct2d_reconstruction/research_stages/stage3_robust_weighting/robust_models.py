"""Train-only robust pair filters and the empirical WEPL noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MAD_TO_SIGMA = 1.482602218505602


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = MAD_TO_SIGMA * np.median(np.abs(values - center), axis=0)
    return center.astype(np.float64), scale.astype(np.float64)


def _training_groups(
    cells: np.ndarray,
    train: np.ndarray,
    grid_size: tuple[int, int],
    minimum_rows: int,
    maximum_radius: int,
) -> list[np.ndarray]:
    train_indices = np.flatnonzero(train)
    cell_count = grid_size[0] * grid_size[1]
    order = np.argsort(cells[train_indices], kind="stable")
    sorted_indices = train_indices[order]
    sorted_cells = cells[sorted_indices]
    boundaries = np.searchsorted(
        sorted_cells, np.arange(cell_count + 1), side="left"
    )
    by_cell = [
        sorted_indices[boundaries[cell] : boundaries[cell + 1]]
        for cell in range(cell_count)
    ]
    groups: list[np.ndarray] = []
    for cell in range(cell_count):
        x = cell % grid_size[0]
        y = cell // grid_size[0]
        selected = by_cell[cell]
        for radius in range(1, maximum_radius + 1):
            if len(selected) >= minimum_rows:
                break
            neighbours = [
                by_cell[nx + y * grid_size[0]]
                for nx in range(max(0, x - radius), min(grid_size[0], x + radius + 1))
            ]
            selected = (
                np.unique(np.concatenate(neighbours))
                if neighbours
                else np.empty(0, dtype=np.int64)
            )
        if len(selected) < minimum_rows:
            selected = train_indices
        groups.append(selected)
    return groups


@dataclass
class FilterModel:
    name: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, name=np.array(self.name), **self.arrays)

    @classmethod
    def load(cls, path, metadata: dict[str, Any] | None = None) -> "FilterModel":
        with np.load(path, allow_pickle=False) as source:
            name = str(source["name"])
            arrays = {key: source[key] for key in source.files if key != "name"}
        return cls(name, arrays, metadata or {})


def fit_filter(
    name: str,
    features: np.ndarray,
    cells: np.ndarray,
    inside: np.ndarray,
    train_partition: np.ndarray,
    config: dict[str, Any],
) -> FilterModel:
    train = np.asarray(inside & train_partition, dtype=bool)
    if np.count_nonzero(train) < int(config["minimum_cell_training_rows"]):
        raise ValueError("insufficient train-only rows for filter fit")
    grid_size = tuple(int(value) for value in config["grid_size"])
    groups = _training_groups(
        cells,
        train,
        grid_size,
        int(config["minimum_cell_training_rows"]),
        int(config["maximum_merge_radius_cells"]),
    )
    global_center, global_scale = robust_location_scale(features[train])
    floor = np.maximum(
        global_scale * float(config["scale_floor_fraction"]), 1.0e-12
    )
    cell_count = grid_size[0] * grid_size[1]
    group_counts = np.array([len(group) for group in groups], dtype=np.int32)

    if name == "baseline_3sigma":
        energy_mean = np.empty(cell_count)
        energy_sigma = np.empty(cell_count)
        angle_sigma = np.empty(cell_count)
        for cell, indices in enumerate(groups):
            values = features[indices]
            energy32 = values[:, 0].astype(np.float32)
            angle_x32 = np.square(values[:, 1]).astype(np.float32)
            angle_y32 = np.square(values[:, 2]).astype(np.float32)
            index = np.zeros(len(values), dtype=np.int64)
            sum_energy = np.zeros(1, dtype=np.float32)
            sum_energy_squared = np.zeros(1, dtype=np.float32)
            sum_angle_squared = np.zeros(1, dtype=np.float32)
            np.add.at(sum_energy, index, energy32)
            np.add.at(sum_energy_squared, index, np.square(energy32))
            np.add.at(sum_angle_squared, index, angle_x32)
            np.add.at(sum_angle_squared, index, angle_y32)
            mean = np.float32(sum_energy[0] / len(values))
            variance = np.float32(
                sum_energy_squared[0] / len(values) - np.square(mean)
            )
            energy_mean[cell] = mean
            energy_sigma[cell] = max(
                float(np.sqrt(max(variance, np.float32(0.0)))), floor[0]
            )
            angle_sigma[cell] = max(
                float(
                    np.sqrt(
                        sum_angle_squared[0] / (2.0 * len(values))
                    )
                ),
                max(floor[1], floor[2]),
            )
        arrays = {
            "energy_mean": energy_mean,
            "energy_sigma": energy_sigma,
            "angle_sigma": angle_sigma,
            "group_counts": group_counts,
        }
    elif name in {"median_mad", "robust_mahalanobis"}:
        center = np.empty((cell_count, 3))
        scale = np.empty((cell_count, 3))
        inverse_covariance = np.empty((cell_count, 3, 3))
        trim_quantile = float(config["covariance_trim_quantile"])
        ridge = float(config["covariance_ridge"])
        for cell, indices in enumerate(groups):
            values = features[indices]
            center[cell], local_scale = robust_location_scale(values)
            scale[cell] = np.maximum(local_scale, floor)
            standardized = (values - center[cell]) / scale[cell]
            norm = np.linalg.norm(standardized, axis=1)
            trimmed = standardized[norm <= np.quantile(norm, trim_quantile)]
            covariance = (
                np.cov(trimmed, rowvar=False)
                if len(trimmed) >= 4
                else np.eye(3, dtype=np.float64)
            )
            covariance = np.atleast_2d(covariance) + ridge * np.eye(3)
            inverse_covariance[cell] = np.linalg.pinv(covariance, hermitian=True)
        arrays = {
            "center": center,
            "scale": scale,
            "inverse_covariance": inverse_covariance,
            "group_counts": group_counts,
        }
    else:
        raise ValueError(f"unsupported filter: {name}")
    return FilterModel(
        name=name,
        arrays={key: np.asarray(value) for key, value in arrays.items()},
        metadata={
            "fit_rows": int(np.count_nonzero(train)),
            "global_center": global_center.tolist(),
            "global_scale": global_scale.tolist(),
            "scale_floor": floor.tolist(),
        },
    )


def apply_filter(
    model: FilterModel,
    features: np.ndarray,
    cells: np.ndarray,
    inside: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.zeros(len(features), dtype=bool)
    distance = np.full(len(features), np.inf, dtype=np.float64)
    indices = np.flatnonzero(inside)
    cell = cells[indices]
    values = features[indices]
    if model.name == "baseline_3sigma":
        energy_z = np.abs(
            (values[:, 0] - model.arrays["energy_mean"][cell])
            / model.arrays["energy_sigma"][cell]
        )
        angle_x_z = np.abs(values[:, 1]) / model.arrays["angle_sigma"][cell]
        angle_y_z = np.abs(values[:, 2]) / model.arrays["angle_sigma"][cell]
        local_distance = np.maximum.reduce((energy_z, angle_x_z, angle_y_z))
        local_selected = local_distance <= float(config["baseline_sigma_cut"])
    else:
        standardized = (
            values - model.arrays["center"][cell]
        ) / model.arrays["scale"][cell]
        if model.name == "median_mad":
            energy_z = np.abs(standardized[:, 0])
            scatter_distance = np.linalg.norm(standardized[:, 1:], axis=1)
            local_distance = np.maximum(energy_z, scatter_distance)
            local_selected = local_distance <= float(config["mad_cut"])
        else:
            inverse = model.arrays["inverse_covariance"][cell]
            local_distance = np.sqrt(
                np.maximum(
                    np.einsum(
                        "ni,nij,nj->n", standardized, inverse, standardized
                    ),
                    0.0,
                )
            )
            local_selected = local_distance <= float(config["mahalanobis_cut"])
    selected[indices] = local_selected
    distance[indices] = local_distance
    return selected, distance


def fit_two_component_gmm(
    standardized: np.ndarray, iterations: int = 30
) -> dict[str, np.ndarray]:
    """Small diagonal two-Gaussian pilot; component 0 is relabelled clean."""

    values = np.asarray(standardized, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1)
    clean = norm <= np.quantile(norm, 0.85)
    means = np.vstack(
        (
            np.mean(values[clean], axis=0),
            np.mean(values[~clean], axis=0),
        )
    )
    variances = np.vstack(
        (
            np.var(values[clean], axis=0) + 1.0e-3,
            np.var(values[~clean], axis=0) + 1.0e-3,
        )
    )
    mixing = np.array([0.85, 0.15], dtype=np.float64)
    for _ in range(iterations):
        log_probability = []
        for component in range(2):
            log_probability.append(
                np.log(max(mixing[component], 1.0e-9))
                - 0.5 * np.sum(np.log(2.0 * np.pi * variances[component]))
                - 0.5
                * np.sum(
                    (values - means[component]) ** 2 / variances[component],
                    axis=1,
                )
            )
        log_probability = np.column_stack(log_probability)
        maximum = np.max(log_probability, axis=1, keepdims=True)
        responsibility = np.exp(log_probability - maximum)
        responsibility /= responsibility.sum(axis=1, keepdims=True)
        mass = responsibility.sum(axis=0) + 1.0e-12
        mixing = mass / len(values)
        means = responsibility.T @ values / mass[:, None]
        for component in range(2):
            difference = values - means[component]
            variances[component] = (
                responsibility[:, component, None] * difference**2
            ).sum(axis=0) / mass[component] + 1.0e-3
    clean_component = int(np.argmin(np.linalg.norm(means, axis=1)))
    return {
        "means": means,
        "variances": variances,
        "mixing": mixing,
        "clean_component": np.array(clean_component, dtype=np.int64),
    }


def gmm_clean_posterior(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    log_probability = []
    for component in range(2):
        variance = model["variances"][component]
        log_probability.append(
            np.log(max(model["mixing"][component], 1.0e-9))
            - 0.5 * np.sum(np.log(2.0 * np.pi * variance))
            - 0.5
            * np.sum(
                (values - model["means"][component]) ** 2 / variance, axis=1
            )
        )
    log_probability = np.column_stack(log_probability)
    maximum = np.max(log_probability, axis=1, keepdims=True)
    probability = np.exp(log_probability - maximum)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability[:, int(model["clean_component"])]


@dataclass
class NoiseModel:
    energy_mev: np.ndarray
    sigma_mm: np.ndarray
    bin_count: np.ndarray
    minimum_sigma_mm: float

    def predict(self, energy_out_mev: np.ndarray) -> np.ndarray:
        predicted = np.exp(
            np.interp(
                energy_out_mev,
                self.energy_mev,
                np.log(self.sigma_mm),
                left=np.log(self.sigma_mm[0]),
                right=np.log(self.sigma_mm[-1]),
            )
        )
        return np.maximum(predicted, self.minimum_sigma_mm)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            energy_mev=self.energy_mev,
            sigma_mm=self.sigma_mm,
            bin_count=self.bin_count,
            minimum_sigma_mm=np.array(self.minimum_sigma_mm),
        )

    @classmethod
    def load(cls, path) -> "NoiseModel":
        with np.load(path, allow_pickle=False) as source:
            return cls(
                source["energy_mev"],
                source["sigma_mm"],
                source["bin_count"],
                float(source["minimum_sigma_mm"]),
            )


def fit_noise_model(
    energy_out_mev: np.ndarray,
    residual_wepl_mm: np.ndarray,
    config: dict[str, Any],
) -> NoiseModel:
    width = float(config["energy_bin_width_mev"])
    bin_index = np.floor(np.asarray(energy_out_mev) / width).astype(np.int64)
    centers, sigmas, counts = [], [], []
    for index in np.unique(bin_index):
        values = np.asarray(residual_wepl_mm)[bin_index == index]
        if len(values) < int(config["minimum_bin_rows"]):
            continue
        center = np.median(values)
        sigma = MAD_TO_SIGMA * np.median(np.abs(values - center))
        centers.append((index + 0.5) * width)
        sigmas.append(max(float(sigma), float(config["minimum_sigma_mm"])))
        counts.append(len(values))
    if len(centers) < 2:
        raise RuntimeError("noise model has fewer than two populated energy bins")
    return NoiseModel(
        np.asarray(centers, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        float(config["minimum_sigma_mm"]),
    )


def normalize_and_clip_weights(
    raw: np.ndarray,
    training_selected: np.ndarray,
    clip: tuple[float, float],
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    reference = raw[training_selected & np.isfinite(raw) & (raw > 0)]
    if len(reference) == 0:
        raise ValueError("cannot normalize weights without selected training rows")
    normalized = raw / np.median(reference)
    normalized[~np.isfinite(normalized)] = clip[0]
    return np.clip(normalized, clip[0], clip[1]).astype(np.float32)
