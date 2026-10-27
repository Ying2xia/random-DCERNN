"""Evaluation and repeated-experiment summaries for composite expectiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value, dtype=np.float64)


def _validated_inputs(
    y_true: object, predictions: object, taus: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_numpy(y_true).reshape(-1)
    pred = _as_numpy(predictions)
    tau_array = np.asarray(tuple(float(tau) for tau in taus), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.ndim != 2:
        raise ValueError("predictions must have shape (n_samples, n_expectiles).")
    if pred.shape != (y.shape[0], tau_array.shape[0]):
        raise ValueError(
            "Prediction shape does not match the response and expectile levels."
        )
    if np.any((tau_array <= 0.0) | (tau_array >= 1.0)):
        raise ValueError("All expectile levels must lie between zero and one.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("Responses and predictions must be finite.")
    return y, pred, tau_array


def _normalized_sample_weight(
    sample_weight: object | None, n_samples: int
) -> np.ndarray:
    if sample_weight is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
    weights = _as_numpy(sample_weight).reshape(-1)
    if weights.shape[0] != n_samples:
        raise ValueError("sample_weight has the wrong length.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain finite nonnegative values.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total weight.")
    return weights / total


def expectile_risk(
    y_true: object,
    predictions: object,
    taus: Iterable[float],
    *,
    sample_weight: object | None = None,
) -> tuple[float, np.ndarray]:
    """Return the equally weighted composite and level-specific risks."""

    y, pred, tau_array = _validated_inputs(y_true, predictions, taus)
    weights = _normalized_sample_weight(sample_weight, y.shape[0])
    residual = y[:, None] - pred
    asymmetric_weight = np.where(
        residual >= 0.0, tau_array[None, :], 1.0 - tau_array[None, :]
    )
    losses = asymmetric_weight * residual * residual
    per_tau = weights @ losses
    return float(np.mean(per_tau)), np.asarray(per_tau, dtype=np.float64)


def expectile_calibration(
    y_true: object,
    predictions: object,
    taus: Iterable[float],
    *,
    sample_weight: object | None = None,
) -> tuple[float, np.ndarray]:
    """Absolute empirical first-order expectile score.

    The population score is zero at a correctly calibrated conditional
    expectile.  The returned aggregate is the mean absolute score across
    levels.
    """

    y, pred, tau_array = _validated_inputs(y_true, predictions, taus)
    weights = _normalized_sample_weight(sample_weight, y.shape[0])
    residual = y[:, None] - pred
    asymmetric_weight = np.where(
        residual >= 0.0, tau_array[None, :], 1.0 - tau_array[None, :]
    )
    score = weights @ (asymmetric_weight * residual)
    absolute_score = np.abs(score)
    return float(np.mean(absolute_score)), absolute_score


def crossing_rate(
    predictions: object,
    *,
    sample_weight: object | None = None,
) -> tuple[float, float]:
    """Return sample-wise and adjacent-pair expectile crossing rates."""

    pred = _as_numpy(predictions)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.ndim != 2:
        raise ValueError("predictions must be a two-dimensional array.")
    if pred.shape[1] < 2:
        return 0.0, 0.0
    weights = _normalized_sample_weight(sample_weight, pred.shape[0])
    crossings = pred[:, :-1] > pred[:, 1:]
    sample_crossing = weights @ np.any(crossings, axis=1)
    pair_crossing = weights @ np.mean(crossings, axis=1)
    return float(sample_crossing), float(pair_crossing)


def true_function_error(
    predictions: object, true_expectiles: object
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """MAE and RMSE against known conditional expectile functions."""

    pred = _as_numpy(predictions)
    truth = _as_numpy(true_expectiles)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("predictions and true_expectiles must have equal 2-D shapes.")
    difference = pred - truth
    per_tau_mae = np.mean(np.abs(difference), axis=0)
    per_tau_rmse = np.sqrt(np.mean(difference * difference, axis=0))
    return (
        float(np.mean(np.abs(difference))),
        float(np.sqrt(np.mean(difference * difference))),
        per_tau_mae,
        per_tau_rmse,
    )


@dataclass(frozen=True)
class RegressionMetrics:
    composite_risk: float
    per_tau_risk: np.ndarray
    calibration: float
    per_tau_calibration: np.ndarray
    crossing_rate: float
    pairwise_crossing_rate: float
    true_mae: float | None = None
    true_rmse: float | None = None
    per_tau_true_mae: np.ndarray | None = None
    per_tau_true_rmse: np.ndarray | None = None

    def to_flat_dict(self, taus: Iterable[float]) -> dict[str, float]:
        tau_values = tuple(float(tau) for tau in taus)
        result: dict[str, float] = {
            "composite_risk": self.composite_risk,
            "calibration": self.calibration,
            "crossing_rate": self.crossing_rate,
            "pairwise_crossing_rate": self.pairwise_crossing_rate,
        }
        for index, tau in enumerate(tau_values):
            suffix = _tau_suffix(tau)
            result[f"risk_tau_{suffix}"] = float(self.per_tau_risk[index])
            result[f"calibration_tau_{suffix}"] = float(
                self.per_tau_calibration[index]
            )
        if self.true_mae is not None:
            result["true_mae"] = self.true_mae
            result["true_rmse"] = float(self.true_rmse)
            assert self.per_tau_true_mae is not None
            assert self.per_tau_true_rmse is not None
            for index, tau in enumerate(tau_values):
                suffix = _tau_suffix(tau)
                result[f"true_mae_tau_{suffix}"] = float(
                    self.per_tau_true_mae[index]
                )
                result[f"true_rmse_tau_{suffix}"] = float(
                    self.per_tau_true_rmse[index]
                )
        return result


def evaluate_predictions(
    y_true: object,
    predictions: object,
    taus: Iterable[float],
    *,
    true_expectiles: object | None = None,
    sample_weight: object | None = None,
) -> RegressionMetrics:
    """Compute every metric used in the simulation and data experiments."""

    tau_values = tuple(float(tau) for tau in taus)
    composite, per_tau = expectile_risk(
        y_true, predictions, tau_values, sample_weight=sample_weight
    )
    calibration, per_tau_calibration = expectile_calibration(
        y_true, predictions, tau_values, sample_weight=sample_weight
    )
    sample_crossing, pair_crossing = crossing_rate(
        predictions,
        sample_weight=sample_weight,
    )
    if true_expectiles is None:
        return RegressionMetrics(
            composite,
            per_tau,
            calibration,
            per_tau_calibration,
            sample_crossing,
            pair_crossing,
        )
    mae, rmse, per_tau_mae, per_tau_rmse = true_function_error(
        predictions, true_expectiles
    )
    return RegressionMetrics(
        composite,
        per_tau,
        calibration,
        per_tau_calibration,
        sample_crossing,
        pair_crossing,
        mae,
        rmse,
        per_tau_mae,
        per_tau_rmse,
    )


def _tau_suffix(tau: float) -> str:
    return f"{tau:.6g}".replace("-", "m").replace(".", "p")


def confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Mean and two-sided Student-t confidence interval."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, np.nan, np.nan
    standard_error = float(stats.sem(array))
    critical = float(
        stats.t.ppf((1.0 + float(confidence)) / 2.0, df=array.size - 1)
    )
    return mean, mean - critical * standard_error, mean + critical * standard_error


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise-error adjustment, preserving original order."""

    p_array = np.asarray(p_values, dtype=np.float64)
    if p_array.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("p_values must lie in [0, 1].")
    order = np.argsort(p_array)
    adjusted = np.empty_like(p_array)
    running = 0.0
    count = p_array.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_array[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_method_tests(
    records: Sequence[Mapping[str, object]],
    *,
    reference_method: str = "FedCERNN",
    metric: str = "composite_risk",
    method_key: str = "method",
    repetition_key: str = "repetition",
) -> list[dict[str, float | str | int]]:
    """Paired t-tests versus one reference with Holm adjustment."""

    by_method: dict[str, dict[int, float]] = {}
    for record in records:
        try:
            method = str(record[method_key])
            repetition = int(record[repetition_key])
            value = float(record[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            by_method.setdefault(method, {})[repetition] = value
    if reference_method not in by_method:
        raise ValueError(f"No records found for reference method {reference_method!r}.")

    comparisons: list[dict[str, float | str | int]] = []
    raw_p_values: list[float] = []
    reference = by_method[reference_method]
    for method in sorted(by_method):
        if method == reference_method:
            continue
        common = sorted(set(reference).intersection(by_method[method]))
        if len(common) < 2:
            continue
        ref_values = np.asarray([reference[index] for index in common])
        other_values = np.asarray([by_method[method][index] for index in common])
        differences = other_values - ref_values
        difference_mean, ci_lower, ci_upper = confidence_interval(differences)
        difference_sd = float(np.std(differences, ddof=1))
        if difference_sd <= np.finfo(np.float64).eps:
            if abs(difference_mean) <= np.finfo(np.float64).eps:
                statistic, p_value = 0.0, 1.0
            else:
                statistic = float(np.sign(difference_mean) * np.inf)
                p_value = 0.0
        else:
            statistic, p_value = stats.ttest_1samp(differences, popmean=0.0)
            if not np.isfinite(p_value):
                statistic, p_value = 0.0, 1.0
        comparisons.append(
            {
                "method": method,
                "reference": reference_method,
                "n_pairs": len(common),
                "mean_difference": difference_mean,
                "difference_ci_lower": ci_lower,
                "difference_ci_upper": ci_upper,
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
        raw_p_values.append(float(p_value))
    if comparisons:
        for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
            comparison["holm_p_value"] = float(adjusted)
    return comparisons
