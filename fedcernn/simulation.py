"""Synthetic location--scale regression experiments for FedCERNN.

The generators in this module deliberately separate the conditional location
and scale from the standardized error distribution.  Consequently, the true
conditional expectile is available as

    mu(x) + sigma(x) * e_tau,

where ``e_tau`` is computed deterministically from the error law rather than
estimated from the generated response sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
from scipy import optimize, special, stats


DEFAULT_TAUS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
SUPPORTED_DGPS: tuple[str, ...] = (
    "smooth_additive",
    "friedman",
    "correlated_interaction",
    "sparse_interaction",
    "correlated_multi_index",
    "hierarchical_composition",
)
SUPPORTED_ERRORS: tuple[str, ...] = (
    "normal",
    "t3",
    "centered_chi2",
    "contaminated_normal",
)


@dataclass(frozen=True)
class SimulationData:
    """A generated regression sample with its population targets."""

    X: np.ndarray
    y: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    true_expectiles: np.ndarray
    taus: tuple[float, ...]
    dgp: str
    error: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _canonical_dgp(name: str) -> str:
    aliases = {
        "additive": "smooth_additive",
        "smooth": "smooth_additive",
        "friedman1": "friedman",
        "correlated": "correlated_interaction",
        "moderate_interaction": "correlated_interaction",
        "sparse": "sparse_interaction",
        "high_dimensional": "sparse_interaction",
    }
    value = aliases.get(str(name).lower(), str(name).lower())
    if value not in SUPPORTED_DGPS:
        raise ValueError(f"Unknown DGP {name!r}; choose from {SUPPORTED_DGPS}.")
    return value


def _canonical_error(name: str) -> str:
    aliases = {
        "gaussian": "normal",
        "student_t3": "t3",
        "student-t3": "t3",
        "chi2": "centered_chi2",
        "chi-square": "centered_chi2",
        "asymmetric_contamination": "contaminated_normal",
        "contaminated_gaussian": "contaminated_normal",
    }
    value = aliases.get(str(name).lower(), str(name).lower())
    if value not in SUPPORTED_ERRORS:
        raise ValueError(
            f"Unknown error distribution {name!r}; choose from {SUPPORTED_ERRORS}."
        )
    return value


def _validate_taus(taus: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(tau) for tau in taus)
    if not values:
        raise ValueError("At least one expectile level is required.")
    if any(not 0.0 < tau < 1.0 for tau in values):
        raise ValueError("Expectile levels must lie strictly between zero and one.")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Expectile levels must be strictly increasing.")
    return values


def _error_cdf_and_lower_moment(
    error: str, value: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return F(z) and E[Z 1{Z<=z}] for a standardized error Z."""

    error = _canonical_error(error)
    z = np.asarray(value, dtype=np.float64)

    if error == "normal":
        cdf = special.ndtr(z)
        lower = -np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        return cdf, lower

    if error == "t3":
        degrees = 3.0
        standardizer = np.sqrt(degrees / (degrees - 2.0))
        raw_value = z * standardizer
        density = stats.t.pdf(raw_value, df=degrees)
        cdf = stats.t.cdf(raw_value, df=degrees)
        raw_lower = -(degrees + raw_value * raw_value) * density / (degrees - 1.0)
        return cdf, raw_lower / standardizer

    if error == "centered_chi2":
        degrees = 3.0
        standardizer = np.sqrt(2.0 * degrees)
        raw_value = np.maximum(degrees + standardizer * z, 0.0)
        cdf = stats.chi2.cdf(raw_value, df=degrees)
        first_raw = degrees * stats.chi2.cdf(raw_value, df=degrees + 2.0)
        lower = (first_raw - degrees * cdf) / standardizer
        below_support = degrees + standardizer * z <= 0.0
        return np.where(below_support, 0.0, cdf), np.where(
            below_support, 0.0, lower
        )

    # A 90/10 mixture of N(0, 1) and N(3, 1), centered and variance-scaled.
    probabilities = np.asarray((0.9, 0.1), dtype=np.float64)
    means = np.asarray((0.0, 3.0), dtype=np.float64)
    mixture_mean = float(probabilities @ means)
    mixture_variance = float(
        probabilities @ (np.ones_like(means) + (means - mixture_mean) ** 2)
    )
    standardizer = np.sqrt(mixture_variance)
    raw_value = mixture_mean + standardizer * z
    shifted = raw_value[..., None] - means
    component_cdf = special.ndtr(shifted)
    component_lower = means * component_cdf - np.exp(
        -0.5 * shifted * shifted
    ) / np.sqrt(2.0 * np.pi)
    cdf = component_cdf @ probabilities
    raw_lower = component_lower @ probabilities
    lower = (raw_lower - mixture_mean * cdf) / standardizer
    return cdf, lower


def _expectile_equation(value: float, tau: float, error: str) -> float:
    cdf_array, lower_array = _error_cdf_and_lower_moment(error, value)
    cdf = float(np.asarray(cdf_array))
    lower = float(np.asarray(lower_array))
    # Every implemented error is centered, hence E[Z 1{Z>e}] = -lower.
    upper_deviation = -lower - value * (1.0 - cdf)
    lower_deviation = value * cdf - lower
    return tau * upper_deviation - (1.0 - tau) * lower_deviation


@lru_cache(maxsize=128)
def _cached_error_expectile(error: str, tau: float) -> float:
    if tau == 0.5:
        return 0.0
    radius = 4.0
    while radius <= 1.0e6:
        lower_value = _expectile_equation(-radius, tau, error)
        upper_value = _expectile_equation(radius, tau, error)
        if lower_value >= 0.0 and upper_value <= 0.0:
            return float(
                optimize.brentq(
                    _expectile_equation,
                    -radius,
                    radius,
                    args=(tau, error),
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
            )
        radius *= 2.0
    raise RuntimeError(f"Could not bracket the {tau:g}-expectile of {error}.")


def error_expectiles(
    error: str, taus: Iterable[float] = DEFAULT_TAUS
) -> np.ndarray:
    """Numerically solve the population expectiles of a standardized error."""

    canonical_error = _canonical_error(error)
    tau_values = _validate_taus(taus)
    return np.asarray(
        [_cached_error_expectile(canonical_error, tau) for tau in tau_values],
        dtype=np.float64,
    )


def sample_standardized_error(
    error: str, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a mean-zero, variance-one error from one of the four study laws."""

    if size < 1:
        raise ValueError("size must be positive.")
    error = _canonical_error(error)
    if error == "normal":
        return rng.standard_normal(size)
    if error == "t3":
        return rng.standard_t(df=3.0, size=size) / np.sqrt(3.0)
    if error == "centered_chi2":
        return (rng.chisquare(df=3.0, size=size) - 3.0) / np.sqrt(6.0)

    component = rng.random(size) >= 0.9
    raw = rng.standard_normal(size) + 3.0 * component.astype(np.float64)
    return (raw - 0.3) / np.sqrt(1.81)


def _smooth_additive(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_features < 6:
        raise ValueError("smooth_additive requires at least six features.")
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    location = (
        2.0 * np.sin(np.pi * X[:, 0])
        + 1.5 * (X[:, 1] ** 2 - 1.0 / 3.0)
        + 0.75 * np.cos(np.pi * X[:, 2])
        + 0.75 * X[:, 3]
        + 0.5 * np.sin(2.0 * np.pi * X[:, 4])
    )
    scale = 0.6 + 0.4 * special.expit(X[:, 0] + X[:, 5])
    return X, location, scale


def _friedman(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_features < 6:
        raise ValueError("friedman requires at least six features.")
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    location = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )
    scale = 0.75 + 0.5 * X[:, 5]
    return X, location, scale


def _correlated_interaction(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moderate-dimensional nonlinear regression with correlated covariates.

    The first ten coordinates carry nonlinear signal; the remaining coordinates
    are noise features.  An AR(1) covariance structure makes this a distinct
    moderate-dimensional setting from the independent Friedman and sparse
    high-dimensional mechanisms.
    """

    if n_features < 20:
        raise ValueError("correlated_interaction requires at least 20 features.")
    indices = np.arange(n_features, dtype=np.float64)
    covariance = 0.45 ** np.abs(indices[:, None] - indices[None, :])
    X = rng.multivariate_normal(
        mean=np.zeros(n_features, dtype=np.float64),
        cov=covariance,
        size=n_samples,
    )
    location = (
        1.8 * np.sin(X[:, 0])
        + 1.4 * X[:, 1] * X[:, 2]
        + 1.1 * np.tanh(X[:, 3] * X[:, 4])
        + 0.9 * (X[:, 5] ** 2 - 1.0)
        + 0.7 * np.sin(X[:, 6] * X[:, 7])
        + 0.5 * X[:, 8]
        - 0.4 * X[:, 9]
    )
    scale = 0.6 + 0.5 * special.expit(0.6 * X[:, 0] - 0.4 * X[:, 1] + X[:, 10])
    return X, location, scale


def _correlated_multi_index(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded correlated inputs with nonlinear effects of shared indices.

    Candidate experiment, fixed before its development runs. The Gaussian
    copula separates dependence from the unbounded quadratic signals in the
    existing correlated-interaction example.
    """
    if n_features < 20:
        raise ValueError("correlated_multi_index requires at least 20 features.")
    indices = np.arange(n_features)
    covariance = 0.5 ** np.abs(indices[:, None] - indices[None, :])
    z = rng.multivariate_normal(np.zeros(n_features), covariance, size=n_samples)
    X = 2.0 * special.ndtr(z) - 1.0
    u = (X[:, 0] + X[:, 1]) / 2.0
    v = (X[:, 2] - X[:, 3]) / 2.0
    location = (
        2.0 * np.sin(np.pi * u)
        + 1.5 * np.tanh(2.0 * v)
        + 0.75 * np.cos(np.pi * X[:, 4])
        + 0.75 * X[:, 5]
        + 0.5 * np.sin(np.pi * X[:, 6] * X[:, 7])
    )
    scale = 0.6 + 0.4 * special.expit(X[:, 0] + X[:, 8])
    return X, location, scale


def _hierarchical_composition(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse hierarchical interactions among bounded high-dimensional inputs.

    Latent averages and products enter an outer nonlinear response function.
    This is a distinct compositional alternative to additive or unbounded
    polynomial interaction mechanisms, not a network-generated target.
    """
    if n_features < 30:
        raise ValueError("hierarchical_composition requires at least 30 features.")
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    u = (X[:, 0] + X[:, 1]) / 2.0
    v = X[:, 2] * X[:, 3]
    w = (X[:, 4] - X[:, 5]) / 2.0
    location = (
        2.0 * np.sin(np.pi * (u + 0.5 * v))
        + 1.5 * (w ** 2 - 1.0 / 6.0)
        + 0.75 * np.cos(np.pi * X[:, 6])
        + 0.5 * X[:, 7] * X[:, 8]
    )
    scale = 0.6 + 0.4 * special.expit(X[:, 0] + X[:, 9])
    return X, location, scale


def _sparse_interaction(
    n_samples: int, rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_features < 30:
        raise ValueError("sparse_interaction requires at least 30 features.")
    X = rng.standard_normal(size=(n_samples, n_features))
    location = (
        2.0 * np.sin(X[:, 0])
        + 1.5 * X[:, 1] * X[:, 2]
        + 1.0 * (X[:, 3] ** 2 - 1.0)
        + 0.75 * X[:, 4] * X[:, 5]
        + 0.5 * X[:, 6]
    )
    scale = 0.6 + 0.6 * special.expit(X[:, 0] + X[:, 1])
    return X, location, scale


_DGP_FUNCTIONS: dict[
    str,
    Callable[
        [int, np.random.Generator, int],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
] = {
    "smooth_additive": _smooth_additive,
    "friedman": _friedman,
    "correlated_interaction": _correlated_interaction,
    "sparse_interaction": _sparse_interaction,
    "correlated_multi_index": _correlated_multi_index,
    "hierarchical_composition": _hierarchical_composition,
}


def generate_simulation(
    n_samples: int,
    dgp: str = "smooth_additive",
    error: str = "normal",
    *,
    seed: int = 0,
    taus: Iterable[float] = DEFAULT_TAUS,
    n_features: int | None = None,
) -> SimulationData:
    """Generate one reproducible synthetic regression sample.

    ``n_features`` defaults to 10 for the first two designs, 20 for the
    correlated designs, and 30 for the sparse or hierarchical designs.
    All arrays use double precision.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    canonical_dgp = _canonical_dgp(dgp)
    canonical_error = _canonical_error(error)
    tau_values = _validate_taus(taus)
    if n_features is None:
        if canonical_dgp in {"sparse_interaction", "hierarchical_composition"}:
            n_features = 30
        elif canonical_dgp in {"correlated_interaction", "correlated_multi_index"}:
            n_features = 20
        else:
            n_features = 10
    n_features = int(n_features)

    covariate_seed, error_seed = np.random.SeedSequence(int(seed)).spawn(2)
    rng = np.random.default_rng(covariate_seed)
    X, location, scale = _DGP_FUNCTIONS[canonical_dgp](
        int(n_samples), rng, n_features
    )
    errors = sample_standardized_error(
        canonical_error, int(n_samples), np.random.default_rng(error_seed)
    )
    y = location + scale * errors
    population_errors = error_expectiles(canonical_error, tau_values)
    truth = location[:, None] + scale[:, None] * population_errors[None, :]
    return SimulationData(
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        location=np.asarray(location, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        true_expectiles=np.asarray(truth, dtype=np.float64),
        taus=tau_values,
        dgp=canonical_dgp,
        error=canonical_error,
    )
