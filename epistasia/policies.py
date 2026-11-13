from __future__ import annotations

from typing import Literal
import numpy as np

NanPolicy = Literal["nanmean", "propagate", "error"]
MissingPolicy = Literal["error", "drop"]

_VALID_NAN_POLICIES = {"nanmean", "propagate", "error"}
_VALID_MISSING_POLICIES = {"error", "drop"}


def normalize_nan_policy(nan_policy: str) -> str:
    """
    Normalize and validate NaN policy.

    Legacy compatibility:
    - 'omit' is accepted as an alias of 'nanmean'.
    """
    if nan_policy == "omit":
        return "nanmean"

    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            "nan_policy must be 'nanmean', 'propagate', or 'error' "
            "('omit' is accepted as a legacy alias of 'nanmean')."
        )
    return nan_policy


def normalize_missing_policy(missing_policy: str) -> str:
    """
    Normalize and validate missing-state policy.

    - 'error': require all states needed by the calculation.
    - 'drop': compute only where the required states are available.
    """
    if missing_policy not in _VALID_MISSING_POLICIES:
        raise ValueError("missing_policy must be 'error' or 'drop'.")
    return missing_policy


def as_2d_values(values: np.ndarray) -> np.ndarray:
    """
    Ensure values has shape (M, R).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must be 1D or 2D.")
    return values


def nanmean_rows(values: np.ndarray) -> np.ndarray:
    """
    Mean over replicate columns, ignoring NaNs.
    Rows with all-NaN replicates become NaN.
    """
    values = as_2d_values(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(values, axis=1)