"""Configuration-driven, resumable FedCERNN experiments."""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .baselines import (
    BASELINE_NAMES,
    BaselineConfig,
    fit_baseline,
    predict,
)
from .data_meps import MEPSSplits, prepare_meps
from .data_eicu import EICUMatrixSplits, load_restricted_eicu_matrices
from .federated import FedCERNN, equal_iid_partition
from .metrics import confidence_interval, evaluate_predictions, paired_method_tests
from .model import CERNN
from .simulation import (
    DEFAULT_TAUS,
    SUPPORTED_DGPS,
    SUPPORTED_ERRORS,
    generate_simulation,
)
from .training import OptimizerConfig


FEDCERNN_METHOD = "FedCERNN"
FEDCERNN_T5_METHOD = "FedCERNN-T5"
ALL_METHODS: tuple[str, ...] = (
    FEDCERNN_METHOD,
    FEDCERNN_T5_METHOD,
) + BASELINE_NAMES

CSV_FIELDS: tuple[str, ...] = (
    "run_key",
    "scenario_id",
    "scenario_kind",
    "dataset",
    "dgp",
    "error",
    "repetition",
    "base_seed",
    "seed",
    "method",
    "status",
    "n_train",
    "n_test",
    "n_clients",
    "rounds",
    "communication_rounds",
    "n_expectiles",
    "hidden_dims",
    "composite_risk",
    "calibration",
    "crossing_rate",
    "pairwise_crossing_rate",
    "true_mae",
    "true_rmse",
    "communication_bytes",
    "elapsed_seconds",
    "metrics_json",
    "history_json",
    "scenario_json",
    "error_message",
    "timestamp_utc",
)


@dataclass(frozen=True)
class ExperimentScenario:
    scenario_id: str
    kind: str
    settings: Mapping[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _scenario_id(kind: str, settings: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        (kind + "\n" + _canonical_json(settings)).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def derive_seed(base_seed: int, scenario_id: str, repetition: int) -> int:
    """Stable seed independent of Python's randomized hash implementation."""

    payload = f"{int(base_seed)}:{scenario_id}:{int(repetition)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _run_key(
    scenario_id: str,
    repetition: int,
    seed: int,
    method: str,
) -> str:
    """Execution identity, including the realized scenario seed."""

    return (
        f"{scenario_id}:{int(repetition)}:{int(seed)}:"
        f"{str(method)}"
    )


def _scenario_seed_key(scenario: ExperimentScenario) -> str:
    """Return a data-only seed identity, independent of method settings.

    The full scenario id deliberately includes optimizer and method settings
    so result records from different configurations cannot collide.  Those
    settings must not, however, change the simulated data or a health-data
    train/validation/test split.  One-factor sensitivities additionally share
    a common stochastic replicate so the changed axis is isolated.
    """

    simulation = scenario.settings.get("simulation", {})
    dgp = str(simulation.get("dgp", ""))
    error = str(simulation.get("error", ""))
    seed_policy = str(scenario.settings.get("seed_policy", "legacy"))
    if seed_policy == "shared_dgp_v1" and dgp and error:
        return f"simulation:{dgp}:{error}"
    if seed_policy == "shared_primary_v1" and dgp and error:
        return f"primary:{dgp}:{error}"
    if seed_policy not in {"legacy", "shared_dgp_v1", "shared_primary_v1"}:
        raise ValueError(f"Unknown seed policy {seed_policy!r}.")
    if scenario.kind.startswith("sensitivity:"):
        return f"sensitivity-reference:{dgp}:{error}"
    if scenario.kind == "primary":
        return f"primary:{dgp}:{error}"
    if scenario.kind in {"meps", "eicu"}:
        return scenario.kind
    return str(scenario.kind)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("The experiment YAML root must be a mapping.")
    loaded["_config_path"] = str(source.resolve())
    return loaded


def _deep_update(
    target: MutableMapping[str, Any], updates: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = copy.deepcopy(value)
    return target


def _set_nested(settings: MutableMapping[str, Any], path: str, value: Any) -> None:
    pieces = path.split(".")
    cursor = settings
    for piece in pieces[:-1]:
        child = cursor.setdefault(piece, {})
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set sensitivity path {path!r}.")
        cursor = child
    cursor[pieces[-1]] = copy.deepcopy(value)


def _base_simulation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed_policy": str(config.get("seed_policy", "legacy")),
        "seed_base_by_dgp": copy.deepcopy(
            config.get("seed_base_by_dgp", {})
        ),
        "simulation": copy.deepcopy(config.get("simulation", {})),
        "model": copy.deepcopy(config.get("model", {})),
        "optimization": copy.deepcopy(config.get("optimization", {})),
        "fedcernn": copy.deepcopy(config.get("fedcernn", {})),
        "baselines": copy.deepcopy(config.get("baselines", {})),
        "participation": copy.deepcopy(config.get("participation", {})),
        "methods": copy.deepcopy(config.get("methods", list(ALL_METHODS))),
    }


def expand_scenarios(config: Mapping[str, Any]) -> list[ExperimentScenario]:
    """Expand all DGP/error pairs plus one-factor sensitivity scenarios."""

    dataset = str(config.get("dataset", "simulation")).lower()
    if dataset != "simulation":
        settings = {
            "seed_policy": str(config.get("seed_policy", "legacy")),
            "dataset": dataset,
            "data": copy.deepcopy(config.get("data", {})),
            "model": copy.deepcopy(config.get("model", {})),
            "optimization": copy.deepcopy(config.get("optimization", {})),
            "fedcernn": copy.deepcopy(config.get("fedcernn", {})),
            "baselines": copy.deepcopy(config.get("baselines", {})),
            "participation": copy.deepcopy(config.get("participation", {})),
            "methods": copy.deepcopy(config.get("methods", list(ALL_METHODS))),
        }
        return [
            ExperimentScenario(
                scenario_id=_scenario_id(dataset, settings),
                kind=dataset,
                settings=settings,
            )
        ]

    template = _base_simulation_settings(config)
    simulation = template["simulation"]
    include_primary = bool(simulation.pop("include_primary", True))
    explicit_pairs = simulation.pop("scenarios", None)
    dgps = simulation.pop("dgps", simulation.get("dgp", ["smooth_additive"]))
    errors = simulation.pop("errors", simulation.get("error", ["normal"]))
    if isinstance(dgps, str):
        dgps = [dgps]
    if isinstance(errors, str):
        errors = [errors]
    if explicit_pairs is not None:
        if not isinstance(explicit_pairs, Sequence) or isinstance(
            explicit_pairs, (str, bytes)
        ):
            raise ValueError("simulation.scenarios must be a list of mappings.")
        pairs: list[tuple[str, str]] = []
        for pair in explicit_pairs:
            if not isinstance(pair, Mapping) or "dgp" not in pair or "error" not in pair:
                raise ValueError(
                    "Every simulation.scenarios item must define dgp and error."
                )
            pairs.append((str(pair["dgp"]), str(pair["error"])))
        if not pairs:
            raise ValueError("simulation.scenarios cannot be empty.")
    else:
        pairs = [(str(dgp), str(error)) for dgp in dgps for error in errors]

    scenarios: list[ExperimentScenario] = []
    if include_primary:
        for dgp, error in pairs:
            settings = copy.deepcopy(template)
            settings["simulation"]["dgp"] = dgp
            settings["simulation"]["error"] = error
            identifier = _scenario_id("primary", settings)
            scenarios.append(ExperimentScenario(identifier, "primary", settings))

    sensitivity = config.get("sensitivity", {})
    if not isinstance(sensitivity, Mapping) or not sensitivity.get("enabled", False):
        return scenarios
    reference = copy.deepcopy(template)
    reference_overrides = sensitivity.get("reference", {})
    if isinstance(reference_overrides, Mapping):
        _deep_update(reference, reference_overrides)
    reference["simulation"]["dgp"] = str(
        reference["simulation"].get("dgp", pairs[0][0])
    )
    reference["simulation"]["error"] = str(
        reference["simulation"].get("error", pairs[0][1])
    )
    axes = sensitivity.get("axes", {})
    if not isinstance(axes, Mapping):
        raise ValueError("sensitivity.axes must be a mapping.")
    seen = {scenario.scenario_id for scenario in scenarios}
    if bool(sensitivity.get("include_reference", False)):
        identifier = _scenario_id("sensitivity:reference", reference)
        if identifier not in seen:
            seen.add(identifier)
            scenarios.append(
                ExperimentScenario(identifier, "sensitivity:reference", reference)
            )
    for path, raw_values in axes.items():
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
            raise ValueError(f"Sensitivity axis {path!r} must be a list.")
        for value in raw_values:
            settings = copy.deepcopy(reference)
            _set_nested(settings, str(path), value)
            kind = f"sensitivity:{path}"
            identifier = _scenario_id(kind, settings)
            if identifier in seen:
                continue
            seen.add(identifier)
            scenarios.append(ExperimentScenario(identifier, kind, settings))
    return scenarios


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    repetitions = int(experiment.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("experiment.repetitions must be positive.")
    scenarios = expand_scenarios(config)
    if not scenarios:
        raise ValueError("The configuration expands to no scenarios.")
    for scenario in scenarios:
        settings = scenario.settings
        methods = settings.get("methods", [])
        if not methods:
            raise ValueError("At least one method must be configured.")
        unknown = set(methods).difference(ALL_METHODS)
        if unknown:
            raise ValueError("Unknown configured methods: " + ", ".join(unknown))
        if str(config.get("dataset", "simulation")).lower() == "simulation":
            simulation = settings["simulation"]
            if simulation["dgp"] not in SUPPORTED_DGPS:
                raise ValueError(f"Unknown DGP {simulation['dgp']!r}.")
            if simulation["error"] not in SUPPORTED_ERRORS:
                raise ValueError(f"Unknown error {simulation['error']!r}.")
            if int(simulation.get("n_train", 0)) < int(
                simulation.get("n_clients", 1)
            ):
                raise ValueError("n_train must be at least n_clients.")


class ResultStore:
    """Append-only JSONL plus a fixed-column CSV view for safe resumption.

    One active writer is allowed per result directory so resumed processes
    cannot append the same run key concurrently.
    """

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_directory / "results.jsonl"
        self.csv_path = self.output_directory / "results.csv"
        self.lock_path = self.output_directory / ".run.lock"
        self._lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.seek(0)
            owner = self._lock_handle.read().strip() or "unknown process"
            self._lock_handle.close()
            raise RuntimeError(
                f"An experiment is already writing to {self.output_directory} "
                f"({owner})."
            ) from error
        self._lock_handle.seek(0)
        self._lock_handle.truncate()
        self._lock_handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
        )
        self._lock_handle.flush()
        os.fsync(self._lock_handle.fileno())
        try:
            self._records = self._read_jsonl()
            self._synchronize_csv()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the exclusive writer lock."""

        if getattr(self, "_lock_handle", None) is None:
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._lock_handle = None

    def _read_jsonl(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line can result from interruption.  It is
                    # ignored and that run will be retried.
                    if line_number == sum(1 for _ in self.jsonl_path.open()):
                        continue
                    raise
                if isinstance(value, dict):
                    records.append(value)
        return records

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @staticmethod
    def _csv_record(record: Mapping[str, Any]) -> dict[str, Any]:
        csv_record = {field: record.get(field, "") for field in CSV_FIELDS}
        for field in ("metrics_json", "history_json", "scenario_json"):
            value = csv_record[field]
            if not isinstance(value, str):
                csv_record[field] = json.dumps(
                    value, ensure_ascii=False, sort_keys=True
                )
        return csv_record

    def _synchronize_csv(self) -> None:
        """Rebuild the CSV view if an interruption left it behind JSONL."""

        if not self._records:
            return
        json_keys = [str(record.get("run_key", "")) for record in self._records]
        csv_keys: list[str] = []
        if self.csv_path.exists():
            try:
                with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) == CSV_FIELDS:
                        csv_keys = [
                            str(row.get("run_key", "")) for row in reader
                        ]
            except (OSError, csv.Error):
                csv_keys = []
        if csv_keys == json_keys:
            return
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self._records:
                writer.writerow(self._csv_record(record))
        os.replace(temporary, self.csv_path)

    def completed_keys(self, *, include_errors: bool = False) -> set[str]:
        keys: set[str] = set()
        for record in self._records:
            if not (include_errors or record.get("status") == "success"):
                continue
            if "run_key" in record:
                keys.add(str(record["run_key"]))
            # Legacy records used scenario:repetition:method as their run key.
            # Their stored realized seed lets us recognize them safely without
            # treating a rerun under a different base seed as completed.
            try:
                keys.add(
                    _run_key(
                        str(record["scenario_id"]),
                        int(record["repetition"]),
                        int(record["seed"]),
                        str(record["method"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
        return keys

    def append(self, record: Mapping[str, Any]) -> None:
        safe_record = _json_safe(dict(record))
        encoded = json.dumps(safe_record, ensure_ascii=False, sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        csv_record = self._csv_record(safe_record)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(csv_record)
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(dict(safe_record))


def _optimizer_config(raw: Mapping[str, Any] | None) -> OptimizerConfig:
    values = dict(raw or {})
    allowed = {
        "adam_steps",
        "adam_lr",
        "lbfgs_steps",
        "lbfgs_lr",
        "tolerance",
        "history_size",
        "stepwise_lbfgs",
        "select_min_gradient_checkpoint",
        "max_rms_parameter_change",
        "project_rms_parameter_change",
    }
    return OptimizerConfig(**{key: values[key] for key in allowed if key in values})


def _baseline_config(settings: Mapping[str, Any]) -> BaselineConfig:
    raw = dict(settings.get("baselines", {}))
    participation = dict(settings.get("participation", {}))
    optimizer = _optimizer_config(settings.get("optimization", {}))
    fedcernn = settings.get("fedcernn", {})
    host_optimizer = _optimizer_config(
        fedcernn.get("host_optimizer", settings.get("optimization", {}))
    )
    local_optimizer = _optimizer_config(raw.get("local_optimizer", {}))
    predictive_raw = raw.get("predictive_local_optimizer")
    predictive_local_optimizer = (
        _optimizer_config(predictive_raw)
        if predictive_raw is not None
        else None
    )
    predictive_by_method = {
        str(method): _optimizer_config(value)
        for method, value in dict(
            raw.get("predictive_local_optimizers", {})
        ).items()
    }
    return BaselineConfig(
        rounds=int(raw.get("rounds", 5)),
        local_steps=int(raw.get("local_steps", 20)),
        local_lr=float(raw.get("local_lr", 1.0e-2)),
        l2_lambda=float(settings.get("optimization", {}).get("l2_lambda", 0.0)),
        fedprox_mu=float(raw.get("fedprox_mu", 1.0e-2)),
        cease_alpha=float(raw.get("cease_alpha", 1.0e-2)),
        feddane_mu=float(raw.get("feddane_mu", 1.0e-2)),
        participation_rate=float(participation.get("rate", 1.0)),
        participation_seed=int(participation.get("seed", 0)),
        centralized_optimizer=optimizer,
        host_optimizer=host_optimizer,
        local_optimizer=local_optimizer,
        predictive_local_optimizer=predictive_local_optimizer,
        predictive_local_optimizers=predictive_by_method,
        corrected_local_optimizers={
            str(method): _optimizer_config(value)
            for method, value in dict(raw.get("corrected_local_optimizers", {})).items()
        },
        scaffold_solver=str(raw.get("scaffold_solver", "legacy")),
        scaffold_sgd_steps=int(raw.get("scaffold_sgd_steps", 100)),
        scaffold_sgd_lr=float(raw.get("scaffold_sgd_lr", 0.03)),
    )


def _make_model(
    input_dim: int,
    settings: Mapping[str, Any],
    taus: Sequence[float],
    seed: int,
) -> CERNN:
    torch.manual_seed(int(seed))
    model_settings = settings.get("model", {})
    hidden_dims = tuple(int(width) for width in model_settings.get("hidden_dims", [32, 16]))
    return CERNN(
        input_dim=int(input_dim),
        hidden_dims=hidden_dims,
        taus=tuple(float(tau) for tau in taus),
        dtype=torch.float64,
    )


def _partition_indices(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    seed: int,
    partition: Mapping[str, Any] | None = None,
) -> list[Tensor]:
    """Create an IID or one-factor heterogeneous client split.

    The global generated sample is held fixed; only its client assignment
    changes.  This isolates the documented source of heterogeneity.
    """

    settings = dict(partition or {})
    kind = str(settings.get("kind", "iid")).lower().replace("-", "_")
    if kind == "iid":
        return equal_iid_partition(len(y), n_clients, seed)
    generator = np.random.default_rng(int(seed))
    if kind == "covariate_shift":
        ordering = np.argsort(np.asarray(X)[:, 0], kind="stable")
    elif kind == "response_shift":
        ordering = np.argsort(np.asarray(y), kind="stable")
    elif kind == "host_shift":
        ordering = np.argsort(np.asarray(X)[:, 0], kind="stable")[::-1]
    elif kind == "size_imbalance":
        concentration = float(settings.get("concentration", 0.3))
        if concentration <= 0.0:
            raise ValueError("partition.concentration must be positive.")
        probabilities = generator.dirichlet(np.full(n_clients, concentration))
        counts = generator.multinomial(len(y), probabilities)
        while np.any(counts == 0):
            probabilities = generator.dirichlet(np.full(n_clients, concentration))
            counts = generator.multinomial(len(y), probabilities)
        shuffled = generator.permutation(len(y))
        offsets = np.cumsum(np.r_[0, counts])
        return [
            torch.as_tensor(shuffled[offsets[index] : offsets[index + 1]])
            for index in range(n_clients)
        ]
    else:
        raise ValueError(
            "partition.kind must be iid, covariate_shift, response_shift, "
            "host_shift, or size_imbalance."
        )
    if kind == "host_shift":
        host_size = int(np.ceil(len(y) / n_clients))
        host = ordering[:host_size]
        tail = generator.permutation(ordering[host_size:])
        return [torch.as_tensor(host.copy())] + [
            torch.as_tensor(piece) for piece in np.array_split(tail, n_clients - 1)
        ]
    return [torch.as_tensor(piece) for piece in np.array_split(ordering, n_clients)]


def _partition_clients(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    seed: int,
    partition: Mapping[str, Any] | None = None,
) -> list[tuple[Tensor, Tensor]]:
    partitions = _partition_indices(X, y, n_clients, seed, partition)
    return [
        (
            torch.as_tensor(X[index.cpu().numpy()], dtype=torch.float64),
            torch.as_tensor(y[index.cpu().numpy()], dtype=torch.float64),
        )
        for index in partitions
    ]


def _fit_method(
    method: str,
    input_dim: int,
    taus: Sequence[float],
    clients: Sequence[tuple[Tensor, Tensor]],
    settings: Mapping[str, Any],
    initialization_seed: int,
) -> tuple[CERNN, int, int, list[dict[str, Any]]]:
    model = _make_model(input_dim, settings, taus, initialization_seed)
    if method in {FEDCERNN_METHOD, FEDCERNN_T5_METHOD}:
        base_optimization = dict(settings.get("optimization", {}))
        fedcernn_settings = settings.get("fedcernn", {})
        surrogate_raw = dict(fedcernn_settings.get("surrogate_optimizer", {}))
        surrogate_raw.pop("profile", None)
        surrogate_optimizer = _optimizer_config(
            {**base_optimization, **surrogate_raw}
        )
        host_raw = fedcernn_settings.get("host_optimizer", base_optimization)
        host_optimizer = _optimizer_config(host_raw)
        trainer = FedCERNN(
            model,
            l2_lambda=float(
                settings.get("optimization", {}).get("l2_lambda", 0.0)
            ),
            optimizer_config=surrogate_optimizer,
            host_optimizer_config=host_optimizer,
            host_regularization_path=fedcernn_settings.get(
                "host_regularization_path"
            ),
        )
        participation = settings.get("participation", {})
        result = trainer.fit(
            clients,
            rounds=(
                5
                if method == FEDCERNN_T5_METHOD
                else int(settings.get("fedcernn", {}).get("rounds", 1))
            ),
            participation_rate=float(participation.get("rate", 1.0)),
            participation_seed=int(participation.get("seed", 0)),
        )
        communication_rounds = sum(
            int(item.get("communication_bytes", 0)) > 0
            for item in result.history
        )
        return (
            result.model,
            result.communication_bytes,
            communication_rounds,
            result.history,
        )
    result = fit_baseline(method, model, clients, _baseline_config(settings))
    communication_rounds = sum(
        int(item.get("communication_bytes", 0)) > 0 for item in result.history
    )
    return (
        result.model,
        result.communication_bytes,
        communication_rounds,
        result.history,
    )


def _fit_real_method(
    method, input_dim, taus, clients, settings, initialization_seed,
    validation_x, validation_y,
):
    """Select a prespecified solver on validation data only, then refit.

    No test data are accepted by this interface. Search costs are recorded
    separately; reported final-fit payload and time exclude hyperparameter
    search, as for the other baselines. All candidates share initialization.
    """
    candidates = settings.get("baselines", {}).get("validation_candidates", {}).get(method, [])
    if not candidates:
        return _fit_method(method, input_dim, taus, clients, settings, initialization_seed), {}
    search_start = time.perf_counter()
    entries = []
    best = None
    for candidate in candidates:
        candidate_settings = copy.deepcopy(settings)
        _deep_update(candidate_settings["baselines"], candidate["baselines"])
        entry = {"name": candidate["name"], "baselines": candidate["baselines"]}
        try:
            fit = _fit_method(method, input_dim, taus, clients, candidate_settings, initialization_seed)
            # Positive-affine scaling preserves the ordering of expectile risks.
            score = evaluate_predictions(validation_y, predict(fit[0], validation_x), taus).composite_risk
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation risk.")
            entry.update(validation_risk_standardized=float(score), communication_bytes=int(fit[1]))
            if best is None or score < best[0]:
                best = (float(score), candidate_settings, candidate["name"])
            del fit
        except (FloatingPointError, RuntimeError) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        entries.append(entry)
    if best is None:
        raise FloatingPointError(f"All validation solver candidates failed for {method}.")
    search_seconds = time.perf_counter() - search_start
    result = _fit_method(method, input_dim, taus, clients, best[1], initialization_seed)
    return result, {
        "selection_metric": "validation_composite_risk_standardized",
        "selected": best[2], "candidates": entries,
        "elapsed_seconds": search_seconds,
        "communication_bytes": sum(e.get("communication_bytes", 0) for e in entries),
        "test_used_for_selection": False,
        "cost_scope": "completed_candidate_fits; failed-fit payload unavailable",
    }


def _record_base(
    scenario: ExperimentScenario,
    repetition: int,
    base_seed: int,
    seed: int,
    method: str,
    *,
    dataset: str,
) -> dict[str, Any]:
    simulation = scenario.settings.get("simulation", {})
    data = scenario.settings.get("data", {})
    n_train = simulation.get("n_train", data.get("n_train", ""))
    n_test = simulation.get("n_test", data.get("n_test", ""))
    n_clients = simulation.get("n_clients", data.get("n_clients", ""))
    rounds = _configured_method_rounds(method, scenario.settings)
    taus = scenario.settings.get("model", {}).get("taus", DEFAULT_TAUS)
    hidden = scenario.settings.get("model", {}).get("hidden_dims", [32, 16])
    run_key = _run_key(scenario.scenario_id, repetition, seed, method)
    return {
        "run_key": run_key,
        "scenario_id": scenario.scenario_id,
        "scenario_kind": scenario.kind,
        "dataset": dataset,
        "dgp": simulation.get("dgp", ""),
        "error": simulation.get("error", ""),
        "repetition": int(repetition),
        "base_seed": int(base_seed),
        "seed": int(seed),
        "method": method,
        "status": "running",
        "n_train": n_train,
        "n_test": n_test,
        "n_clients": n_clients,
        "rounds": rounds,
        "communication_rounds": "",
        "n_expectiles": len(taus),
        "hidden_dims": "-".join(str(width) for width in hidden),
        "scenario_json": scenario.settings,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _configured_method_rounds(
    method: str,
    settings: Mapping[str, Any],
) -> int:
    """Return the configured number of network communication rounds."""

    if method in {"Intercept-only", "Centralized CERNN", "Host CERNN"}:
        return 0
    if method == FEDCERNN_METHOD:
        return int(settings.get("fedcernn", {}).get("rounds", 1))
    if method == FEDCERNN_T5_METHOD:
        return 5
    if method == "one-shot FedAvg":
        return 1
    baseline_settings = settings.get("baselines", {})
    baseline_rounds = int(baseline_settings.get("rounds", 5))
    if method == "DCERNN":
        return baseline_rounds * int(baseline_settings.get("local_steps", 20))
    return baseline_rounds


def _run_simulation_repetition(
    scenario: ExperimentScenario,
    repetition: int,
    base_seed: int,
    seed: int,
    store: ResultStore,
    completed: set[str],
    *,
    fail_fast: bool,
) -> None:
    settings = scenario.settings
    simulation = settings["simulation"]
    taus = tuple(
        float(tau)
        for tau in settings.get("model", {}).get("taus", DEFAULT_TAUS)
    )
    train = generate_simulation(
        int(simulation["n_train"]),
        str(simulation["dgp"]),
        str(simulation["error"]),
        seed=seed,
        taus=taus,
        n_features=simulation.get("n_features"),
    )
    test = generate_simulation(
        int(simulation.get("n_test", 5000)),
        str(simulation["dgp"]),
        str(simulation["error"]),
        seed=(seed + 1) % (2**32),
        taus=taus,
        n_features=train.n_features,
    )
    clients = _partition_clients(
        train.X,
        train.y,
        int(simulation.get("n_clients", 20)),
        seed=(seed + 2) % (2**32),
        partition=simulation.get("partition"),
    )
    initialization_seed = (seed + 3) % (2**32)

    for method in settings["methods"]:
        record = _record_base(
            scenario,
            repetition,
            base_seed,
            seed,
            method,
            dataset="simulation",
        )
        if record["run_key"] in completed:
            continue
        started = time.perf_counter()
        try:
            model, communication, communication_rounds, history = _fit_method(
                method,
                train.n_features,
                taus,
                clients,
                settings,
                initialization_seed,
            )
            predictions = predict(model, test.X)
            metrics = evaluate_predictions(
                test.y,
                predictions,
                taus,
                true_expectiles=test.true_expectiles,
            ).to_flat_dict(taus)
            record.update(metrics)
            record.update(
                {
                    "status": "success",
                    "communication_bytes": int(communication),
                    "communication_rounds": int(communication_rounds),
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": metrics,
                    "history_json": history,
                    "error_message": "",
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "communication_bytes": "",
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": {},
                    "history_json": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
            store.append(record)
            if fail_fast:
                raise
            continue
        store.append(record)


def _run_meps_repetition(
    scenario: ExperimentScenario,
    repetition: int,
    base_seed: int,
    seed: int,
    store: ResultStore,
    completed: set[str],
    *,
    fail_fast: bool,
) -> None:
    settings = scenario.settings
    data_settings = settings["data"]
    path = data_settings.get("path")
    if not path:
        raise ValueError(
            "data.path must point to the locally downloaded MEPS HC-251 file."
        )
    splits: MEPSSplits = prepare_meps(
        path,
        seed=seed,
        feature_columns=data_settings.get("feature_columns"),
        target_transform=str(data_settings.get("target_transform", "raw")),
    )
    n_clients = int(data_settings.get("n_clients", 10))
    clients = splits.clients(n_clients, seed=(seed + 1) % (2**32))
    taus = tuple(
        float(tau)
        for tau in settings.get("model", {}).get("taus", DEFAULT_TAUS)
    )
    initialization_seed = (seed + 2) % (2**32)
    for method in settings["methods"]:
        record = _record_base(
            scenario,
            repetition,
            base_seed,
            seed,
            method,
            dataset="meps",
        )
        record["n_train"] = len(splits.y_train)
        record["n_test"] = len(splits.y_test)
        record["n_clients"] = n_clients
        if record["run_key"] in completed:
            continue
        started = time.perf_counter()
        try:
            fit, tuning = _fit_real_method(
                method,
                splits.X_train.shape[1],
                taus,
                clients,
                settings,
                initialization_seed,
                splits.X_validation, splits.y_validation,
            )
            model, communication, communication_rounds, history = fit
            started += tuning.get("elapsed_seconds", 0.0)
            record["tuning_json"] = tuning
            standardized_predictions = predict(model, splits.X_test)
            if splits.target_transform == "raw":
                evaluation_target = splits.y_test_thousands
                evaluation_predictions = splits.inverse_target(
                    standardized_predictions
                )
            else:
                # Expectiles are positive-affine equivariant but not equivariant
                # under expm1.  The log1p sensitivity is therefore evaluated on
                # its modeled log scale, without mislabelling a back-transform
                # as a dollar-scale expectile.
                evaluation_target = np.log1p(splits.y_test_thousands)
                evaluation_predictions = splits.inverse_standardization(
                    standardized_predictions
                )
            metrics = evaluate_predictions(
                evaluation_target,
                evaluation_predictions,
                taus,
            ).to_flat_dict(taus)
            weighted_metrics = evaluate_predictions(
                evaluation_target,
                evaluation_predictions,
                taus,
                sample_weight=splits.test_weights,
            ).to_flat_dict(taus)
            metrics.update(
                {
                    f"survey_weighted_{key}": value
                    for key, value in weighted_metrics.items()
                }
            )
            standardized_validation_predictions = predict(model, splits.X_validation)
            if splits.target_transform == "raw":
                validation_target = splits.y_validation_thousands
                validation_predictions = splits.inverse_target(
                    standardized_validation_predictions
                )
            else:
                validation_target = np.log1p(splits.y_validation_thousands)
                validation_predictions = splits.inverse_standardization(
                    standardized_validation_predictions
                )
            validation_metrics = evaluate_predictions(
                validation_target, validation_predictions, taus
            ).to_flat_dict(taus)
            metrics.update(
                {
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                }
            )
            record.update(metrics)
            record.update(
                {
                    "status": "success",
                    "communication_bytes": int(communication),
                    "communication_rounds": int(communication_rounds),
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": metrics,
                    "history_json": history,
                    "error_message": "",
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": {},
                    "history_json": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
            store.append(record)
            if fail_fast:
                raise
            continue
        store.append(record)


def _run_eicu_repetition(
    scenario: ExperimentScenario,
    repetition: int,
    base_seed: int,
    seed: int,
    store: ResultStore,
    completed: set[str],
    *,
    fail_fast: bool,
) -> None:
    """Evaluate methods on identifier-free, protected local eICU matrices."""

    settings = scenario.settings
    data_settings = settings["data"]
    matrix_path = data_settings.get("matrix_path")
    if not matrix_path:
        raise ValueError(
            "data.matrix_path must point to the local protected eICU matrices."
        )
    splits: EICUMatrixSplits = load_restricted_eicu_matrices(matrix_path)
    n_clients = int(data_settings.get("n_clients", 20))
    clients = splits.clients(n_clients, seed=(seed + 1) % (2**32))
    taus = tuple(
        float(tau)
        for tau in settings.get("model", {}).get("taus", DEFAULT_TAUS)
    )
    initialization_seed = (seed + 2) % (2**32)
    for method in settings["methods"]:
        record = _record_base(
            scenario, repetition, base_seed, seed, method, dataset="eicu"
        )
        record["n_train"] = len(splits.y_train)
        record["n_test"] = len(splits.y_test)
        record["n_clients"] = n_clients
        if record["run_key"] in completed:
            continue
        started = time.perf_counter()
        try:
            fit, tuning = _fit_real_method(
                method,
                splits.X_train.shape[1],
                taus,
                clients,
                settings,
                initialization_seed,
                splits.X_validation, splits.y_validation,
            )
            model, communication, communication_rounds, history = fit
            started += tuning.get("elapsed_seconds", 0.0)
            record["tuning_json"] = tuning
            predictions = splits.inverse_target(predict(model, splits.X_test))
            target = splits.inverse_target(splits.y_test)
            metrics = evaluate_predictions(target, predictions, taus).to_flat_dict(taus)
            validation_predictions = splits.inverse_target(
                predict(model, splits.X_validation)
            )
            validation_target = splits.inverse_target(splits.y_validation)
            validation_metrics = evaluate_predictions(
                validation_target, validation_predictions, taus
            ).to_flat_dict(taus)
            metrics.update(
                {
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                }
            )
            record.update(metrics)
            record.update(
                {
                    "status": "success",
                    "communication_bytes": int(communication),
                    "communication_rounds": int(communication_rounds),
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": metrics,
                    "history_json": history,
                    "error_message": "",
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics_json": {},
                    "history_json": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
            store.append(record)
            if fail_fast:
                raise
            continue
        store.append(record)


def write_summaries(store: ResultStore) -> None:
    successful = [
        record for record in store.records if record.get("status") == "success"
    ]
    if not successful:
        return
    frame = pd.DataFrame(successful)
    if "base_seed" not in frame:
        frame["base_seed"] = "legacy"
    else:
        frame["base_seed"] = frame["base_seed"].fillna("legacy")
    reported_metric_names = {
        str(metric)
        for record in successful
        for metric in record.get("metrics_json", {})
    }
    reported_metric_names.update(
        {"communication_bytes", "communication_rounds", "elapsed_seconds"}
    )
    numeric_metrics = [
        metric for metric in sorted(reported_metric_names) if metric in frame
    ]
    summary_rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(
        ["scenario_id", "base_seed", "scenario_kind", "method"], dropna=False
    ):
        scenario_id, base_seed, scenario_kind, method = group_key
        for metric in numeric_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                np.float64
            )
            values = values[np.isfinite(values)]
            mean, ci_lower, ci_upper = confidence_interval(values)
            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "base_seed": base_seed,
                    "scenario_kind": scenario_kind,
                    "method": method,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": mean,
                    "std": (
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else np.nan
                    ),
                    "ci_lower_95": ci_lower,
                    "ci_upper_95": ci_upper,
                }
            )
    summary = pd.DataFrame(summary_rows)
    temporary = store.output_directory / "summary.csv.tmp"
    summary.to_csv(temporary, index=False)
    os.replace(temporary, store.output_directory / "summary.csv")

    tests: list[dict[str, Any]] = []
    for (scenario_id, base_seed), records in frame.groupby(
        ["scenario_id", "base_seed"],
        dropna=False,
    ):
        try:
            comparisons = paired_method_tests(
                records.to_dict(orient="records"),
                reference_method=FEDCERNN_METHOD,
            )
        except ValueError:
            continue
        for comparison in comparisons:
            comparison["scenario_id"] = scenario_id
            comparison["base_seed"] = base_seed
            tests.append(comparison)
    test_path = store.output_directory / "paired_tests.json"
    temporary_test_path = test_path.with_suffix(".json.tmp")
    with temporary_test_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(tests), handle, ensure_ascii=False, indent=2)
    os.replace(temporary_test_path, test_path)


def run_experiment(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    output_directory: str | Path | None = None,
    methods: Iterable[str] | None = None,
    max_repetitions: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run or resume a configured experiment and return all stored records."""

    config = (
        load_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else copy.deepcopy(dict(config_or_path))
    )
    selected_methods: tuple[str, ...] | None = None
    if methods is not None:
        selected_methods = tuple(str(method) for method in methods)
        unknown = set(selected_methods).difference(ALL_METHODS)
        if unknown:
            raise ValueError("Unknown selected methods: " + ", ".join(unknown))
    validate_config(config)
    scenarios = expand_scenarios(config)
    if dry_run:
        return [
            {
                "scenario_id": scenario.scenario_id,
                "scenario_kind": scenario.kind,
                "settings": _json_safe(scenario.settings),
            }
            for scenario in scenarios
        ]

    experiment = config.get("experiment", {})
    repetitions = int(experiment.get("repetitions", 1))
    if max_repetitions is not None:
        repetitions = min(repetitions, int(max_repetitions))
    base_seed = int(experiment.get("base_seed", 20260727))
    if int(experiment.get("torch_threads", 1)) > 0:
        torch.set_num_threads(int(experiment.get("torch_threads", 1)))
    fail_fast = bool(experiment.get("fail_fast", False))
    retry_errors = bool(experiment.get("retry_errors", False))
    destination = output_directory or experiment.get("output_directory", "results")
    store = ResultStore(destination)
    try:
        completed = store.completed_keys(include_errors=not retry_errors)
        dataset = str(config.get("dataset", "simulation")).lower()

        for scenario in scenarios:
            # CLI-selected methods override methods copied into each scenario.
            if selected_methods is not None:
                scenario = ExperimentScenario(
                    scenario.scenario_id,
                    scenario.kind,
                    {**scenario.settings, "methods": list(selected_methods)},
                )
            scenario_base_seed = int(
                scenario.settings.get("seed_base_by_dgp", {}).get(
                    str(scenario.settings.get("simulation", {}).get("dgp", "")),
                    base_seed,
                )
            )
            for repetition in range(repetitions):
                if experiment.get("progress", False):
                    print(f"[{dataset}] repetition {repetition + 1}/{repetitions} (completed records are skipped)", flush=True)
                seed = derive_seed(
                    scenario_base_seed,
                    _scenario_seed_key(scenario),
                    repetition,
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if dataset == "simulation":
                    _run_simulation_repetition(
                        scenario,
                        repetition,
                        scenario_base_seed,
                        seed,
                        store,
                        completed,
                        fail_fast=fail_fast,
                    )
                elif dataset == "meps":
                    _run_meps_repetition(
                        scenario,
                        repetition,
                        scenario_base_seed,
                        seed,
                        store,
                        completed,
                        fail_fast=fail_fast,
                    )
                elif dataset == "eicu":
                    _run_eicu_repetition(
                        scenario,
                        repetition,
                        scenario_base_seed,
                        seed,
                        store,
                        completed,
                        fail_fast=fail_fast,
                    )
                else:
                    raise ValueError(f"Unsupported dataset {dataset!r}.")
        write_summaries(store)
        return store.records
    finally:
        store.close()
