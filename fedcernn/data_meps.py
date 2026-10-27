"""Leakage-conscious preparation of the public MEPS 2023 HC-251 file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .compat import zip_strict
from .federated import equal_iid_partition


TARGET_COLUMN = "TOTEXP23"
PERSON_ID_COLUMN = "DUPERSID"
WEIGHT_COLUMN = "PERWT23F"

# Only pre-specified demographics, socioeconomic variables, insurance, general
# health, limitations, and chronic conditions are eligible.  In particular, no
# expenditure component, payer amount, event count, visit count, prescription
# count, or inpatient-day variable can enter through this list.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "AGE23X",
    "EDUCYR",
    "FAMSZE31",
    "FAMINC23",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "SEX",
    "RACETHX",
    "MARRY31X",
    "REGION31",
    "EMPST31",
    "INSCOV23",
)
ORDINAL_FEATURES: tuple[str, ...] = (
    "POVCAT23",
    "RTHLTH31",
    "MNHLTH31",
)
INDICATOR_FEATURES: tuple[str, ...] = (
    "WLKLIM31",
    "ACTLIM31",
    "SOCLIM31",
    "COGLIM31",
    "IADLHP31",
    "ADLHLP31",
    "HIBPDX",
    "CHDDX",
    "ANGIDX",
    "MIDX",
    "OHRTDX",
    "STRKDX",
    "EMPHDX",
    "CHOLDX",
    "CANCERDX",
    "DIABDX_M18",
    "ARTHDX",
    "ASTHDX",
)
FEATURE_WHITELIST: frozenset[str] = frozenset(
    CONTINUOUS_FEATURES
    + CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
    + INDICATOR_FEATURES
)


@dataclass(frozen=True)
class MEPSSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_weights: np.ndarray
    validation_weights: np.ndarray
    test_weights: np.ndarray
    train_person_ids: np.ndarray
    validation_person_ids: np.ndarray
    test_person_ids: np.ndarray
    feature_names: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    target_center: float
    target_scale: float
    target_transform: str
    y_train_thousands: np.ndarray
    y_validation_thousands: np.ndarray
    y_test_thousands: np.ndarray

    def inverse_standardization(self, standardized: np.ndarray) -> np.ndarray:
        """Undo only the training-set affine target standardization."""

        return (
            np.asarray(standardized, dtype=np.float64) * self.target_scale
            + self.target_center
        )

    def inverse_target(self, standardized: np.ndarray) -> np.ndarray:
        """Map standardized predictions back to the analysis outcome scale."""

        transformed = self.inverse_standardization(standardized)
        if self.target_transform == "raw":
            return transformed
        return np.expm1(transformed)

    def client_indices(
        self, n_clients: int = 10, *, seed: int = 0
    ) -> list[np.ndarray]:
        partitions = equal_iid_partition(len(self.y_train), n_clients, seed)
        return [partition.cpu().numpy() for partition in partitions]

    def clients(
        self, n_clients: int = 10, *, seed: int = 0
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.as_tensor(self.X_train[index], dtype=torch.float64),
                torch.as_tensor(self.y_train[index], dtype=torch.float64),
            )
            for index in self.client_indices(n_clients, seed=seed)
        ]


def read_meps(
    path: str | Path, *, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Read a downloaded HC-251 CSV, Parquet, XPORT, or SAS data file."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"MEPS file not found at {source}. Download HC-251 from AHRQ first."
        )
    lower_name = source.name.lower()
    selected = (
        None
        if columns is None
        else {str(column).strip().upper() for column in columns}
    )
    if lower_name.endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        frame = pd.read_csv(
            source,
            usecols=(
                None
                if selected is None
                else lambda column: str(column).strip().upper() in selected
            ),
        )
    elif lower_name.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(
            source, columns=None if selected is None else sorted(selected)
        )
    elif lower_name.endswith(".dta"):
        frame = pd.read_stata(
            source,
            columns=None if selected is None else sorted(selected),
            convert_categoricals=False,
        )
    elif lower_name.endswith((".xpt", ".xport")):
        frame = pd.read_sas(source, format="xport", encoding="utf-8")
    elif lower_name.endswith(".sas7bdat"):
        frame = pd.read_sas(source, format="sas7bdat", encoding="utf-8")
    else:
        raise ValueError(
            "Unsupported MEPS format. Use CSV(.gz), Parquet, DTA, XPT, or SAS7BDAT."
        )
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def validate_meps_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Require every requested feature to belong to the locked whitelist."""

    normalized = tuple(str(column).strip().upper() for column in columns)
    if not normalized:
        raise ValueError("At least one MEPS feature is required.")
    duplicates = sorted({column for column in normalized if normalized.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate MEPS feature columns: {duplicates}.")
    forbidden = sorted(set(normalized).difference(FEATURE_WHITELIST))
    if forbidden:
        raise ValueError(
            "The following variables are not in the no-leakage MEPS whitelist: "
            + ", ".join(forbidden)
        )
    return normalized


def _split_people(
    person_ids: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_people = np.unique(person_ids.astype(str))
    if unique_people.size < 3:
        raise ValueError("At least three distinct people are needed for a split.")
    rng = np.random.default_rng(int(seed))
    shuffled = unique_people.copy()
    rng.shuffle(shuffled)
    n_people = shuffled.size
    n_train = max(1, int(np.floor(0.70 * n_people)))
    n_validation = max(1, int(np.floor(0.15 * n_people)))
    if n_train + n_validation >= n_people:
        n_train = n_people - 2
        n_validation = 1
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_validation],
        shuffled[n_train + n_validation :],
    )


def _continuous_block(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    train_blocks: list[np.ndarray] = []
    validation_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        validation_values = pd.to_numeric(
            validation[column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        test_values = pd.to_numeric(test[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        # Negative values are MEPS missing/inapplicable codes for these fields.
        # Family income is the exception: legitimately negative income is kept.
        if column != "FAMINC23":
            train_values[train_values < 0.0] = np.nan
            validation_values[validation_values < 0.0] = np.nan
            test_values[test_values < 0.0] = np.nan
        else:
            train_values = np.sign(train_values) * np.log1p(np.abs(train_values))
            validation_values = np.sign(validation_values) * np.log1p(
                np.abs(validation_values)
            )
            test_values = np.sign(test_values) * np.log1p(np.abs(test_values))
        train_missing = ~np.isfinite(train_values)
        validation_missing = ~np.isfinite(validation_values)
        test_missing = ~np.isfinite(test_values)
        finite_train = train_values[np.isfinite(train_values)]
        imputation = float(np.median(finite_train)) if finite_train.size else 0.0
        train_values = np.where(np.isfinite(train_values), train_values, imputation)
        validation_values = np.where(
            np.isfinite(validation_values), validation_values, imputation
        )
        test_values = np.where(np.isfinite(test_values), test_values, imputation)
        center = float(np.mean(train_values))
        scale = float(np.std(train_values))
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = 1.0
        train_blocks.append(((train_values - center) / scale)[:, None])
        validation_blocks.append(((validation_values - center) / scale)[:, None])
        test_blocks.append(((test_values - center) / scale)[:, None])
        names.append(column)
        if column != "FAMINC23":
            train_blocks.append(train_missing.astype(np.float64)[:, None])
            validation_blocks.append(validation_missing.astype(np.float64)[:, None])
            test_blocks.append(test_missing.astype(np.float64)[:, None])
            names.append(f"{column}__missing")
    return train_blocks, validation_blocks, test_blocks, names


def _ordinal_block(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    train_blocks: list[np.ndarray] = []
    validation_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        arrays = [
            pd.to_numeric(part[column], errors="coerce").to_numpy(np.float64)
            for part in (train, validation, test)
        ]
        for array in arrays:
            array[array < 0.0] = np.nan
        missing_arrays = [~np.isfinite(array) for array in arrays]
        finite_train = arrays[0][np.isfinite(arrays[0])]
        imputation = float(np.median(finite_train)) if finite_train.size else 0.0
        arrays = [
            np.where(np.isfinite(array), array, imputation) for array in arrays
        ]
        center = float(np.mean(arrays[0]))
        scale = float(np.std(arrays[0]))
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = 1.0
        for blocks, array, missing in zip_strict(
            (train_blocks, validation_blocks, test_blocks),
            arrays,
            missing_arrays,
        ):
            blocks.append(((array - center) / scale)[:, None])
            blocks.append(missing.astype(np.float64)[:, None])
        names.extend((column, f"{column}__missing"))
    return train_blocks, validation_blocks, test_blocks, names


def _indicator_block(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    train_blocks: list[np.ndarray] = []
    validation_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        raw_arrays = [
            pd.to_numeric(part[column], errors="coerce").to_numpy(np.float64)
            for part in (train, validation, test)
        ]
        arrays = [
            np.where(raw == 1.0, 1.0, np.where(raw == 2.0, 0.0, np.nan))
            for raw in raw_arrays
        ]
        missing_arrays = [~np.isfinite(array) for array in arrays]
        finite_train = arrays[0][np.isfinite(arrays[0])]
        if finite_train.size:
            ones = int(np.sum(finite_train == 1.0))
            zeros = int(np.sum(finite_train == 0.0))
            mode = 1.0 if ones > zeros else 0.0
        else:
            mode = 0.0
        arrays = [
            np.where(np.isfinite(array), array, mode) for array in arrays
        ]
        for blocks, array, missing in zip_strict(
            (train_blocks, validation_blocks, test_blocks),
            arrays,
            missing_arrays,
        ):
            blocks.append(array[:, None])
            blocks.append(missing.astype(np.float64)[:, None])
        names.extend((column, f"{column}__missing"))
    return train_blocks, validation_blocks, test_blocks, names


def _categorical_block(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    train_blocks: list[np.ndarray] = []
    validation_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        def clean(values: pd.Series) -> np.ndarray:
            numeric = pd.to_numeric(values, errors="coerce")
            text = values.fillna("__MISSING__").astype(str).str.strip()
            finite_nonnegative = numeric.notna() & numeric.ge(0)
            canonical_numeric = numeric.map(
                lambda value: f"{float(value):g}" if pd.notna(value) else ""
            )
            text = text.mask(finite_nonnegative, canonical_numeric)
            invalid_code = numeric.lt(0) | text.str.match(r"^-\d")
            return text.mask(invalid_code, "__MISSING__").to_numpy()

        train_values = clean(train[column])
        validation_values = clean(validation[column])
        test_values = clean(test[column])
        observed = sorted(
            value
            for value in set(train_values.tolist())
            if value != "__MISSING__"
        )
        categories = tuple(observed + ["__MISSING__"])
        validation_values = np.where(
            np.isin(validation_values, observed),
            validation_values,
            "__MISSING__",
        )
        test_values = np.where(
            np.isin(test_values, observed), test_values, "__MISSING__"
        )
        for category in categories:
            train_blocks.append((train_values == category).astype(np.float64)[:, None])
            validation_blocks.append(
                (validation_values == category).astype(np.float64)[:, None]
            )
            test_blocks.append((test_values == category).astype(np.float64)[:, None])
            names.append(f"{column}={category}")
    return train_blocks, validation_blocks, test_blocks, names


def _encode_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    continuous = [column for column in features if column in CONTINUOUS_FEATURES]
    categorical = [column for column in features if column in CATEGORICAL_FEATURES]
    ordinal = [column for column in features if column in ORDINAL_FEATURES]
    indicators = [column for column in features if column in INDICATOR_FEATURES]
    continuous_parts = _continuous_block(
        train, validation, test, continuous
    )
    categorical_parts = _categorical_block(
        train, validation, test, categorical
    )
    ordinal_parts = _ordinal_block(train, validation, test, ordinal)
    indicator_parts = _indicator_block(train, validation, test, indicators)
    train_blocks = (
        continuous_parts[0]
        + categorical_parts[0]
        + ordinal_parts[0]
        + indicator_parts[0]
    )
    validation_blocks = (
        continuous_parts[1]
        + categorical_parts[1]
        + ordinal_parts[1]
        + indicator_parts[1]
    )
    test_blocks = (
        continuous_parts[2]
        + categorical_parts[2]
        + ordinal_parts[2]
        + indicator_parts[2]
    )
    names = tuple(
        continuous_parts[3]
        + categorical_parts[3]
        + ordinal_parts[3]
        + indicator_parts[3]
    )
    if not train_blocks:
        raise ValueError("No usable MEPS features remain.")
    return (
        np.concatenate(train_blocks, axis=1),
        np.concatenate(validation_blocks, axis=1),
        np.concatenate(test_blocks, axis=1),
        names,
    )


def prepare_meps(
    path_or_frame: str | Path | pd.DataFrame,
    *,
    seed: int = 2023,
    feature_columns: Iterable[str] | None = None,
    target_transform: str = "raw",
) -> MEPSSplits:
    """Prepare MEPS records using a person-level 70/15/15 split.

    The outcome is first expressed in thousands of dollars.  ``target_transform``
    may be ``"raw"`` (the primary analysis) or ``"log1p"`` (sensitivity
    analysis); either transformed target is standardized using training data
    only.
    """

    if isinstance(path_or_frame, pd.DataFrame):
        frame = path_or_frame.copy()
        frame.columns = [str(column).strip().upper() for column in frame.columns]
    else:
        requested_columns = (
            tuple(feature_columns)
            if feature_columns is not None
            else (
                CONTINUOUS_FEATURES
                + CATEGORICAL_FEATURES
                + ORDINAL_FEATURES
                + INDICATOR_FEATURES
            )
        )
        frame = read_meps(
            path_or_frame,
            columns=(
                TARGET_COLUMN,
                PERSON_ID_COLUMN,
                WEIGHT_COLUMN,
                "AGE23X",
                *requested_columns,
            ),
        )
    required = {TARGET_COLUMN, PERSON_ID_COLUMN, WEIGHT_COLUMN, "AGE23X"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("MEPS file is missing required fields: " + ", ".join(missing))

    if feature_columns is None:
        features = tuple(
            column
            for column in (
                CONTINUOUS_FEATURES
                + CATEGORICAL_FEATURES
                + ORDINAL_FEATURES
                + INDICATOR_FEATURES
            )
            if column in frame.columns
        )
        features = validate_meps_feature_columns(features)
    else:
        features = validate_meps_feature_columns(feature_columns)
        unavailable = sorted(set(features).difference(frame.columns))
        if unavailable:
            raise ValueError(
                "Requested MEPS fields are absent: " + ", ".join(unavailable)
            )

    expenditure = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    survey_weight = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce")
    eligible = (
        expenditure.ge(0)
        & expenditure.notna()
        & np.isfinite(expenditure)
        & frame[PERSON_ID_COLUMN].notna()
    )
    cohort = frame.loc[eligible].copy()
    cohort[WEIGHT_COLUMN] = np.where(
        np.isfinite(survey_weight.loc[eligible])
        & survey_weight.loc[eligible].ge(0),
        survey_weight.loc[eligible],
        0.0,
    )
    if len(cohort) < 3:
        raise ValueError("Fewer than three eligible MEPS records remain.")

    train_people, validation_people, test_people = _split_people(
        cohort[PERSON_ID_COLUMN].astype(str).to_numpy(), seed
    )
    person_values = cohort[PERSON_ID_COLUMN].astype(str)
    train = cohort.loc[person_values.isin(train_people)].copy()
    validation = cohort.loc[person_values.isin(validation_people)].copy()
    test = cohort.loc[person_values.isin(test_people)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("The person-level MEPS split produced an empty subset.")

    X_train, X_validation, X_test, encoded_names = _encode_features(
        train, validation, test, features
    )
    y_train_thousands = (
        pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(np.float64)
        / 1000.0
    )
    y_validation_thousands = (
        pd.to_numeric(validation[TARGET_COLUMN], errors="raise").to_numpy(np.float64)
        / 1000.0
    )
    y_test_thousands = (
        pd.to_numeric(test[TARGET_COLUMN], errors="raise").to_numpy(np.float64)
        / 1000.0
    )
    transform = str(target_transform).lower()
    if transform == "raw":
        train_target = y_train_thousands
        validation_target = y_validation_thousands
        test_target = y_test_thousands
    elif transform == "log1p":
        train_target = np.log1p(y_train_thousands)
        validation_target = np.log1p(y_validation_thousands)
        test_target = np.log1p(y_test_thousands)
    else:
        raise ValueError("target_transform must be 'raw' or 'log1p'.")
    target_center = float(np.mean(train_target))
    target_scale = float(np.std(train_target))
    if target_scale < 1.0e-12:
        target_scale = 1.0

    return MEPSSplits(
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=np.asarray(
            (train_target - target_center) / target_scale, dtype=np.float64
        ),
        X_validation=np.asarray(X_validation, dtype=np.float64),
        y_validation=np.asarray(
            (validation_target - target_center) / target_scale, dtype=np.float64
        ),
        X_test=np.asarray(X_test, dtype=np.float64),
        y_test=np.asarray(
            (test_target - target_center) / target_scale, dtype=np.float64
        ),
        train_weights=pd.to_numeric(
            train[WEIGHT_COLUMN], errors="raise"
        ).to_numpy(np.float64),
        validation_weights=pd.to_numeric(
            validation[WEIGHT_COLUMN], errors="raise"
        ).to_numpy(np.float64),
        test_weights=pd.to_numeric(
            test[WEIGHT_COLUMN], errors="raise"
        ).to_numpy(np.float64),
        train_person_ids=train[PERSON_ID_COLUMN].astype(str).to_numpy(),
        validation_person_ids=validation[PERSON_ID_COLUMN].astype(str).to_numpy(),
        test_person_ids=test[PERSON_ID_COLUMN].astype(str).to_numpy(),
        feature_names=encoded_names,
        source_feature_names=features,
        target_center=target_center,
        target_scale=target_scale,
        target_transform=transform,
        y_train_thousands=y_train_thousands,
        y_validation_thousands=y_validation_thousands,
        y_test_thousands=y_test_thousands,
    )
