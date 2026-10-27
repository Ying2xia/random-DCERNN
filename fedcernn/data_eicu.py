"""Validated, authorization-dependent eICU-CRD preprocessing skeleton.

No eICU data are distributed with this repository.  The builder operates only
on a user's local, credentialed PhysioNet download and enforces the 24-hour
feature window before any event aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import stat
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .federated import equal_iid_partition


PATIENT_COLUMNS: frozenset[str] = frozenset(
    {
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "unitvisitnumber",
        "unitdischargeoffset",
        "age",
        "gender",
        "ethnicity",
        "admissionheight",
        "admissionweight",
        "unitadmitsource",
        "apacheadmissiondx",
        "unittype",
    }
)


@dataclass(frozen=True)
class EventTableSpec:
    table: str
    offset_column: str
    feature_column: str | None
    value_columns: tuple[str, ...]
    id_column: str = "patientunitstayid"

    @property
    def required_columns(self) -> frozenset[str]:
        values = {self.id_column, self.offset_column, *self.value_columns}
        if self.feature_column is not None:
            values.add(self.feature_column)
        return frozenset(values)


EVENT_TABLES: Mapping[str, EventTableSpec] = {
    "lab": EventTableSpec(
        table="lab",
        offset_column="labresultoffset",
        feature_column="labname",
        value_columns=("labresult",),
    ),
    "nursecharting": EventTableSpec(
        table="nursecharting",
        offset_column="nursingchartoffset",
        feature_column="nursingchartcelltypevalname",
        value_columns=("nursingchartvalue",),
    ),
    "vitalperiodic": EventTableSpec(
        table="vitalperiodic",
        offset_column="observationoffset",
        feature_column=None,
        value_columns=(
            "heartrate",
            "respiration",
            "sao2",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
            "temperature",
        ),
    ),
}


@dataclass(frozen=True)
class EICUValidationReport:
    root: Path
    files: Mapping[str, Path]
    missing_optional_tables: tuple[str, ...]


@dataclass(frozen=True)
class EICUCohortSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def assert_disjoint_patients(self) -> None:
        groups = [
            set(part["uniquepid"].astype(str))
            for part in (self.train, self.validation, self.test)
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("eICU patient identifiers overlap across splits.")


@dataclass(frozen=True)
class EICUMatrixSplits:
    """Identifier-free, protected matrices for the local eICU analysis."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_center: float
    target_scale: float

    def client_indices(self, n_clients: int, *, seed: int) -> list[np.ndarray]:
        partitions = equal_iid_partition(len(self.y_train), n_clients, seed)
        return [partition.cpu().numpy() for partition in partitions]

    def clients(
        self, n_clients: int, *, seed: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.as_tensor(self.X_train[index], dtype=torch.float64),
                torch.as_tensor(self.y_train[index], dtype=torch.float64),
            )
            for index in self.client_indices(n_clients, seed=seed)
        ]

    def inverse_target(self, standardized: np.ndarray) -> np.ndarray:
        return (
            np.asarray(standardized, dtype=np.float64) * self.target_scale
            + self.target_center
        )


def load_restricted_eicu_matrices(path: str | Path) -> EICUMatrixSplits:
    """Load the identifier-free eICU matrices from a protected local file."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Protected eICU feature matrices were not found at {source}."
        )
    if stat.S_IMODE(source.stat().st_mode) & 0o077:
        raise PermissionError(
            "Protected eICU matrices must not be readable by group or other users."
        )
    required = {
        "X_train",
        "X_validation",
        "X_test",
        "y_train",
        "y_validation",
        "y_test",
        "feature_names",
        "target_center",
        "target_scale",
    }
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("Protected eICU matrix archive has an unexpected schema.")
        prohibited = {"patient", "stay", "identifier"}
        if any(token in " ".join(archive.files).lower() for token in prohibited):
            raise ValueError("Protected eICU matrix archive must not contain identifiers.")
        values = {name: archive[name].copy() for name in required}
    X_train = np.asarray(values["X_train"], dtype=np.float64)
    X_validation = np.asarray(values["X_validation"], dtype=np.float64)
    X_test = np.asarray(values["X_test"], dtype=np.float64)
    y_train = np.asarray(values["y_train"], dtype=np.float64)
    y_validation = np.asarray(values["y_validation"], dtype=np.float64)
    y_test = np.asarray(values["y_test"], dtype=np.float64)
    matrices = (X_train, X_validation, X_test)
    targets = (y_train, y_validation, y_test)
    if (
        any(matrix.ndim != 2 or not np.isfinite(matrix).all() for matrix in matrices)
        or any(target.ndim != 1 or not np.isfinite(target).all() for target in targets)
        or any(len(matrix) != len(target) for matrix, target in zip(matrices, targets))
        or len({matrix.shape[1] for matrix in matrices}) != 1
    ):
        raise ValueError("Protected eICU matrices are invalid or not finite.")
    names = tuple(str(name) for name in values["feature_names"].tolist())
    if len(names) != X_train.shape[1]:
        raise ValueError("eICU feature-name count does not match matrix width.")
    center = float(np.asarray(values["target_center"], dtype=np.float64).reshape(-1)[0])
    scale = float(np.asarray(values["target_scale"], dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("eICU target standardization metadata is invalid.")
    return EICUMatrixSplits(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_names=names,
        target_center=center,
        target_scale=scale,
    )


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in frame.columns]
    return result


def _find_table(root: Path, table: str) -> Path | None:
    candidates = (
        root / f"{table}.csv.gz",
        root / f"{table}.csv",
        root / f"{table}.parquet",
    )
    direct = next((path for path in candidates if path.exists()), None)
    if direct is not None:
        return direct
    # The canonical eICU release uses camel-case filenames such as
    # ``vitalPeriodic.csv.gz``.  Match filenames case-insensitively so the
    # same restricted-local pipeline works on case-sensitive filesystems.
    expected = {path.name.lower() for path in candidates}
    return next(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.name.lower() in expected
        ),
        None,
    )


def _read_header(path: Path) -> tuple[str, ...]:
    if path.suffix.lower() == ".parquet":
        return tuple(str(column).lower() for column in pd.read_parquet(path).columns)
    return tuple(
        str(column).lower() for column in pd.read_csv(path, nrows=0).columns
    )


def validate_eicu_directory(
    root: str | Path,
    *,
    require_event_tables: bool = False,
    required_event_tables: Sequence[str] | None = None,
) -> EICUValidationReport:
    """Validate table presence and fields without reading protected records.

    ``required_event_tables`` permits a study to validate only the explicitly
    selected source tables.  This is important for restricted local analyses:
    a missing, unused table must not force its extraction or processing.
    ``require_event_tables=True`` retains the former all-event-table check.
    """

    directory = Path(root).expanduser()
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"eICU directory not found at {directory}. "
            "Obtain access through PhysioNet before running this pipeline."
        )
    patient_path = _find_table(directory, "patient")
    if patient_path is None:
        raise FileNotFoundError("patient.csv(.gz) or patient.parquet is required.")
    patient_header = set(_read_header(patient_path))
    missing_patient = sorted(PATIENT_COLUMNS.difference(patient_header))
    if missing_patient:
        raise ValueError(
            "The eICU patient table is missing fields: "
            + ", ".join(missing_patient)
        )

    requested = (
        set(EVENT_TABLES)
        if require_event_tables
        else {
            str(table).strip().lower()
            for table in (required_event_tables or ())
        }
    )
    unknown = sorted(requested.difference(EVENT_TABLES))
    if unknown:
        raise ValueError("Unknown requested eICU feature tables: " + ", ".join(unknown))

    files: dict[str, Path] = {"patient": patient_path}
    missing_optional: list[str] = []
    for table, spec in EVENT_TABLES.items():
        path = _find_table(directory, table)
        if path is None:
            missing_optional.append(table)
            continue
        missing_fields = sorted(spec.required_columns.difference(_read_header(path)))
        if missing_fields:
            raise ValueError(
                f"The eICU {table} table is missing fields: "
                + ", ".join(missing_fields)
            )
        files[table] = path
    missing_requested = sorted(requested.intersection(missing_optional))
    if missing_requested:
        raise FileNotFoundError(
            "Missing requested eICU feature tables: " + ", ".join(missing_requested)
        )
    return EICUValidationReport(
        root=directory,
        files=files,
        missing_optional_tables=tuple(missing_optional),
    )


def _read_table(path: Path, **kwargs: object) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=kwargs.get("usecols"))
    else:
        frame = pd.read_csv(path, **kwargs)
    return _normalize_columns(frame)


def _parse_age(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    # eICU masks ages over 89 as the literal string "> 89".
    masked_old = text.str.contains(r">\s*89", regex=True, na=False)
    numeric = pd.to_numeric(text, errors="coerce")
    return numeric.mask(masked_old, 90.0)


class EICU24HourBuilder:
    """Build the first ICU unit stay within each hospitalization.

    eICU identifies the order of ICU unit visits inside a hospital encounter
    through ``unitvisitnumber``.  It does not expose a reliable absolute
    chronology across separate hospital encounters for the same
    ``uniquepid``.  We therefore retain the first unit stay of each
    ``patienthealthsystemstayid`` and keep all eligible encounters; downstream
    splitting remains grouped by ``uniquepid``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        feature_window_minutes: int = 24 * 60,
        target_cap_days: float | None = None,
    ) -> None:
        if feature_window_minutes <= 0:
            raise ValueError("feature_window_minutes must be positive.")
        if target_cap_days is not None and target_cap_days <= 0.0:
            raise ValueError("target_cap_days must be positive.")
        self.feature_window_minutes = int(feature_window_minutes)
        self.target_cap_days = (
            None if target_cap_days is None else float(target_cap_days)
        )
        self.report = validate_eicu_directory(root)

    def build_cohort(self) -> pd.DataFrame:
        # Read only the allow-listed analytic fields. In particular, no
        # organization-level field is loaded or retained by this pipeline.
        patient = _read_table(
            self.report.files["patient"], usecols=list(PATIENT_COLUMNS)
        )
        patient["age_numeric"] = _parse_age(patient["age"])
        for column in (
            "unitvisitnumber",
            "unitdischargeoffset",
            "patientunitstayid",
            "patienthealthsystemstayid",
        ):
            patient[column] = pd.to_numeric(patient[column], errors="coerce")
        patient = patient.loc[
            patient["uniquepid"].notna()
            & patient["patienthealthsystemstayid"].notna()
            & patient["patientunitstayid"].notna()
            & patient["unitvisitnumber"].notna()
        ].copy()
        patient = patient.sort_values(
            [
                "patienthealthsystemstayid",
                "unitvisitnumber",
                "patientunitstayid",
            ],
            kind="mergesort",
        )
        # The order is identifiable within a hospitalization.  We deliberately
        # do not order separate hospital encounters by opaque identifiers.
        patient = patient.drop_duplicates(
            subset=["patienthealthsystemstayid"], keep="first"
        )
        patient = patient.loc[
            patient["age_numeric"].ge(18)
            & patient["unitdischargeoffset"].ge(self.feature_window_minutes)
        ].copy()
        remaining_days = (
            patient["unitdischargeoffset"].to_numpy(dtype=np.float64)
            - self.feature_window_minutes
        ) / (24.0 * 60.0)
        if self.target_cap_days is not None:
            remaining_days = np.minimum(remaining_days, self.target_cap_days)
        patient["remaining_icu_los_days"] = remaining_days
        patient["feature_window_minutes"] = self.feature_window_minutes
        return patient.reset_index(drop=True)

    def iter_24h_events(
        self,
        table: str,
        *,
        stay_ids: Sequence[int] | np.ndarray | None = None,
        chunksize: int = 250_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield events whose recorded offset lies in [0, 24 hours]."""

        name = str(table).lower()
        if name not in EVENT_TABLES:
            raise ValueError(f"Unknown event table {table!r}.")
        if name not in self.report.files:
            raise FileNotFoundError(
                f"{name}.csv(.gz) was not found in the authorized download."
            )
        spec = EVENT_TABLES[name]
        path = self.report.files[name]
        allowed_stays = (
            None
            if stay_ids is None
            else set(np.asarray(stay_ids, dtype=np.int64).tolist())
        )
        if path.suffix.lower() == ".parquet":
            chunks: Sequence[pd.DataFrame] = [
                _read_table(path, usecols=list(spec.required_columns))
            ]
        else:
            chunks = pd.read_csv(
                path,
                usecols=lambda column: str(column).lower() in spec.required_columns,
                chunksize=int(chunksize),
            )
        for raw_chunk in chunks:
            chunk = _normalize_columns(raw_chunk)
            offsets = pd.to_numeric(chunk[spec.offset_column], errors="coerce")
            ids = pd.to_numeric(chunk[spec.id_column], errors="coerce")
            keep = offsets.between(0, self.feature_window_minutes) & ids.notna()
            if allowed_stays is not None:
                keep &= ids.astype("Int64").isin(allowed_stays)
            filtered = chunk.loc[keep].copy()
            if filtered.empty:
                continue
            filtered[spec.offset_column] = offsets.loc[keep].astype(np.int64)
            filtered[spec.id_column] = ids.loc[keep].astype(np.int64)
            yield filtered

    def aggregate_24h_numeric_events(
        self,
        table: str,
        *,
        stay_ids: Sequence[int] | np.ndarray,
        selected_features: Sequence[str] | None = None,
        chunksize: int = 250_000,
    ) -> pd.DataFrame:
        """Aggregate selected early measurements by mean/min/max/last.

        Dynamic-name tables (lab and nursecharting) require an explicit feature
        list so the analysis cannot silently change when a database version or
        local extraction differs.
        """

        name = str(table).lower()
        if name not in EVENT_TABLES:
            raise ValueError(f"Unknown event table {table!r}.")
        spec = EVENT_TABLES[name]
        if spec.feature_column is not None and not selected_features:
            raise ValueError(
                f"selected_features is required when aggregating {name}."
            )
        selected = (
            None
            if selected_features is None
            else {str(feature).strip().lower() for feature in selected_features}
        )
        # A dense, fixed-size accumulator is deliberately used instead of a
        # Python dictionary keyed by every observed event.  The latter becomes
        # unnecessarily memory-intensive for credentialed eICU event tables.
        # Its rows are restricted to the eligible cohort and its columns to
        # the pre-specified feature panel, so no data-dependent feature search
        # occurs here.
        unique_stays = pd.Index(
            np.asarray(stay_ids, dtype=np.int64)
        ).drop_duplicates().to_numpy(dtype=np.int64)
        if unique_stays.size == 0:
            return pd.DataFrame(index=pd.Index([], name="patientunitstayid"))
        if spec.feature_column is None:
            feature_names = tuple(str(value).lower() for value in spec.value_columns)
        else:
            feature_names = tuple(sorted(selected or set()))
        feature_index = {feature: index for index, feature in enumerate(feature_names)}
        stay_index = {int(stay): index for index, stay in enumerate(unique_stays)}
        shape = (unique_stays.size, len(feature_names))
        counts = np.zeros(shape, dtype=np.float64)
        sums = np.zeros(shape, dtype=np.float64)
        minima = np.full(shape, np.inf, dtype=np.float64)
        maxima = np.full(shape, -np.inf, dtype=np.float64)
        last_offsets = np.full(shape, -np.inf, dtype=np.float64)
        last_values = np.full(shape, np.nan, dtype=np.float64)

        def update(
            stay: np.ndarray,
            feature: np.ndarray,
            offset: np.ndarray,
            value: np.ndarray,
        ) -> None:
            valid = np.isfinite(value)
            if not np.any(valid):
                return
            row = stay[valid]
            column = feature[valid]
            event_offset = offset[valid]
            event_value = value[valid]
            np.add.at(counts, (row, column), 1.0)
            np.add.at(sums, (row, column), event_value)
            np.minimum.at(minima, (row, column), event_value)
            np.maximum.at(maxima, (row, column), event_value)
            keys = row * len(feature_names) + column
            order = np.lexsort((event_offset, keys))
            ordered_keys = keys[order]
            final = order[np.r_[ordered_keys[1:] != ordered_keys[:-1], True]]
            final_row = row[final]
            final_column = column[final]
            later = event_offset[final] >= last_offsets[final_row, final_column]
            if np.any(later):
                last_offsets[final_row[later], final_column[later]] = event_offset[
                    final[later]
                ]
                last_values[final_row[later], final_column[later]] = event_value[
                    final[later]
                ]

        for chunk in self.iter_24h_events(
            name, stay_ids=stay_ids, chunksize=chunksize
        ):
            if spec.feature_column is None:
                for value_column in spec.value_columns:
                    if selected is not None and value_column.lower() not in selected:
                        continue
                    positions = chunk[spec.id_column].map(stay_index)
                    update(
                        positions.to_numpy(dtype=np.int64),
                        np.full(len(chunk), feature_index[value_column.lower()], dtype=np.int64),
                        chunk[spec.offset_column].to_numpy(dtype=np.float64),
                        pd.to_numeric(chunk[value_column], errors="coerce").to_numpy(
                            dtype=np.float64
                        ),
                    )
            else:
                value_column = spec.value_columns[0]
                feature_values = (
                    chunk[spec.feature_column].astype(str).str.strip().str.lower()
                )
                keep = feature_values.isin(selected or set())
                if not np.any(keep):
                    continue
                positions = chunk.loc[keep, spec.id_column].map(stay_index)
                features = feature_values.loc[keep].map(feature_index)
                update(
                    positions.to_numpy(dtype=np.int64),
                    features.to_numpy(dtype=np.int64),
                    chunk.loc[keep, spec.offset_column].to_numpy(dtype=np.float64),
                    pd.to_numeric(
                        chunk.loc[keep, value_column], errors="coerce"
                    ).to_numpy(dtype=np.float64),
                )
        active = np.flatnonzero(np.any(counts > 0.0, axis=0))
        if active.size == 0:
            return pd.DataFrame(index=pd.Index([], name="patientunitstayid"))
        output: dict[str, np.ndarray] = {}
        for index in active:
            feature = feature_names[int(index)]
            observed = counts[:, index] > 0.0
            output[f"{name}__{feature}__mean"] = np.where(
                observed, sums[:, index] / np.maximum(counts[:, index], 1.0), np.nan
            )
            output[f"{name}__{feature}__minimum"] = np.where(
                observed, minima[:, index], np.nan
            )
            output[f"{name}__{feature}__maximum"] = np.where(
                observed, maxima[:, index], np.nan
            )
            output[f"{name}__{feature}__last"] = np.where(
                observed, last_values[:, index], np.nan
            )
            output[f"{name}__{feature}__count"] = counts[:, index]
        return pd.DataFrame(
            output,
            index=pd.Index(unique_stays, name="patientunitstayid"),
        ).sort_index(axis=1)


def split_eicu_patients(
    cohort: pd.DataFrame, *, seed: int = 2024
) -> EICUCohortSplit:
    """Split by deidentified patient identifier, never by ICU stay."""

    required = {"uniquepid", "patientunitstayid", "remaining_icu_los_days"}
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise ValueError("Cohort is missing fields: " + ", ".join(missing))
    people = np.unique(cohort["uniquepid"].astype(str).to_numpy())
    if people.size < 3:
        raise ValueError("At least three distinct eICU patients are required.")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(people)
    n_train = max(1, int(np.floor(0.70 * people.size)))
    n_validation = max(1, int(np.floor(0.15 * people.size)))
    if n_train + n_validation >= people.size:
        n_train, n_validation = people.size - 2, 1
    train_ids = set(people[:n_train])
    validation_ids = set(people[n_train : n_train + n_validation])
    person = cohort["uniquepid"].astype(str)
    result = EICUCohortSplit(
        train=cohort.loc[person.isin(train_ids)].copy(),
        validation=cohort.loc[person.isin(validation_ids)].copy(),
        test=cohort.loc[~person.isin(train_ids | validation_ids)].copy(),
    )
    result.assert_disjoint_patients()
    return result
